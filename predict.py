"""
숨표 CO2 예측 배치 (EC2에서 10분마다 실행)

하는 일
1. 강의실(type=LECTURE)이면서 센서가 있는 방(FULL·SIM)마다
   최근 7일 측정값으로 그 방의 CO2 변화 모델을 학습한다.
     수업 중:   ΔC(5분) = a − b·(C − 바깥)        바깥 = 그 방 최근 7일 최저치(2% 지점)     a = 사람이 내뿜는 양, b = 문틈·환기
     밤(19~07시): ΔC(5분) = −k·(C − 바깥)
     낮에 수업 없음: 지금 값 유지
   (시간표·휴강은 DB의 class_schedule, class_cancellation 사용)
2. 지금 값에서 시작해 "창문을 닫아둔 채로" 120분 뒤까지 5분 간격으로 예측 → prediction 테이블
3. 앞으로 30분 안에 기준(1,000ppm)을 넘으면 PENDING 조치(창문 열기)를 만든다 → action 테이블
   이미 넘었으면 "지금" 조치, 상황이 풀린 조치는 EXPIRED
4. 3일 지난 예측은 지운다

실행
  python predict.py                 # DB 읽고 쓰기 (cron)
  python predict.py --dry-run       # DB 읽기만, 쓰지 않고 결과 출력
  python predict.py --backtest readings.csv [room_id]   # 과거 데이터로 30분 예측 오차(MAE) 확인

DB 접속: 환경변수 DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD (기본 127.0.0.1:5432/sumpyo)
"""
import os
import sys
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
MODEL_VERSION = "greybox-v1"
STEP_MIN = 5
HORIZON_MIN = 120
ALERT_WINDOW_MIN = 30          # 이 안에 기준을 넘으면 조치 생성
THRESHOLD = int(os.getenv("CO2_THRESHOLD", "1000"))
C_OUT = 420                    # 바깥 CO2 기본값 (방마다 최근 최저치로 다시 잡음: 센서 오차 보정)
FIT_DAYS = 7
KEEP_PRED_DAYS = 3

# 데이터가 부족할 때 쓰는 기본값 (5분당)
DEFAULT_K = 0.02               # 빈 강의실 감쇠 (5분당, 약 0.24회/시간 = 617호 실측)
K_RANGE = (0.004, 0.06)        # 문 닫힌 강의실에서 물리적으로 가능한 범위 (약 0.05~0.7회/시간)
MIN_EXCESS = 60                # 바깥보다 이만큼은 높아야 감쇠율 학습에 씀 (낮으면 센서 잡음에 묻힘)
MIN_CLASS_POINTS = 24          # 수업 중 5분 데이터가 이만큼은 있어야 a, b를 학습값으로 씀
MIN_IDLE_POINTS = 48


def now_kst():
    if os.getenv("PREDICT_NOW"):                  # 테스트용: 특정 시각 기준으로 실행
        return datetime.fromisoformat(os.environ["PREDICT_NOW"])
    return datetime.now(KST).replace(tzinfo=None, second=0, microsecond=0)


# ---------------------------------------------------------------- 시간표
class Schedule:
    def __init__(self, sched_df, cancel_df):
        # sched_df: id, room_id, day_of_week(1=월), start_time, end_time
        self.by_room = {}
        for r in sched_df.itertuples():
            s = _to_min(r.start_time)
            e = _to_min(r.end_time)
            self.by_room.setdefault(r.room_id, []).append((r.id, int(r.day_of_week), s, e))
        # cancel_df: class_schedule_id, cancel_date
        self.cancelled = {(int(c.class_schedule_id), pd.Timestamp(c.cancel_date).date())
                          for c in cancel_df.itertuples()}

    def class_at(self, room_id, ts):
        """ts 시각에 진행 중인 수업 id (없으면 None)"""
        dow = ts.weekday() + 1
        t = ts.hour * 60 + ts.minute
        for sid, d, s, e in self.by_room.get(room_id, []):
            if d == dow and s <= t < e and (sid, ts.date()) not in self.cancelled:
                return sid
        return None


def _to_min(v):
    if isinstance(v, str):
        h, m = v.split(":")[:2]
        return int(h) * 60 + int(m)
    return v.hour * 60 + v.minute     # datetime.time


# ---------------------------------------------------------------- 모델
def resample(readings):
    """1분 측정 → 5분 평균 (센서 오작동값 제거)"""
    s = readings[(readings["co2"] > 300) & (readings["co2"] < 4900)]
    s = s.set_index("measured_at").sort_index()
    return s[["co2", "temp", "humidity"]].resample(f"{STEP_MIN}min").mean().interpolate(limit=3)


def fit_room(series, room_id, schedule, exclude=()):
    """5분 단위 CO2로 a, b, k 학습. exclude: 창문 열린 구간 [(시작, 끝)]"""
    c_out = float(np.clip(series.quantile(0.02), 350, 450)) if len(series) >= MIN_IDLE_POINTS else C_OUT
    df = pd.DataFrame({"c": series})
    df["dc"] = df["c"].shift(-1) - df["c"]
    df["ex"] = df["c"] - c_out
    df["cls"] = [schedule.class_at(room_id, t) is not None for t in df.index]
    for s, e in exclude:
        df = df[(df.index < s) | (df.index > e)]
    df = df.dropna()

    # 수업 데이터가 부족하면 a=None → 수업 중엔 "지금 값 유지"로 예측 (모르는 걸 지어내지 않음)
    a, b, k, src = None, 0.0, DEFAULT_K, "default"
    # 감쇠율은 사람이 드나들지 않는 밤(19~07시) 데이터로만 학습 (낮엔 수업 외에도 사람이 있어서 섞임)
    night = (df.index.hour >= 19) | (df.index.hour < 7)
    idle = df[(~df["cls"]) & night & (df["ex"] >= MIN_EXCESS)]
    if len(idle) >= MIN_IDLE_POINTS:
        k = float(np.clip(-(idle["dc"] * idle["ex"]).sum() / (idle["ex"] ** 2).sum(), *K_RANGE))
        src = "fit"
    cls = df[df["cls"]]
    if len(cls) >= MIN_CLASS_POINTS:
        A = np.vstack([np.ones(len(cls)), -cls["ex"]]).T
        a_fit, b_fit = np.linalg.lstsq(A, cls["dc"], rcond=None)[0]
        a = float(max(a_fit, 0.0))           # 수업 중에 CO2가 안 오르는 방이면 0
        b = float(np.clip(b_fit, 0.0, 0.3))
        src = "fit"
    return {"a": a, "b": b, "k": k, "c_out": c_out, "src": src, "n_class": len(cls), "n_idle": len(idle)}


def rollout(c0, start, room_id, params, schedule, horizon=HORIZON_MIN):
    """창문 닫힌 상태로 start부터 horizon분 뒤까지 5분 간격 예측"""
    out, c, t = [], c0, start
    for _ in range(horizon // STEP_MIN):
        if schedule.class_at(room_id, t) is not None:
            if params["a"] is not None:
                c = c + params["a"] - params["b"] * (c - params["c_out"])
        elif t.hour >= 19 or t.hour < 7:
            c = c - params["k"] * (c - params["c_out"])
        # 낮에 수업이 없는 시간은 "지금 값 유지" (빈 시간에도 사람이 드나들어서 감쇠를 가정하면 오히려 틀림: 617호 백테스트)
        t = t + timedelta(minutes=STEP_MIN)
        out.append((t, max(c, 380.0)))
    return out


def first_over(preds, limit_t):
    for t, c in preds:
        if t > limit_t:
            break
        if c >= THRESHOLD:
            return t, c
    return None


# ---------------------------------------------------------------- DB
def connect():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"), port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "sumpyo"), user=os.getenv("DB_USER", "sumpyo"),
        password=os.getenv("DB_PASSWORD", ""))


def load(conn, now):
    def q(sql, p=None):
        with conn.cursor() as cur:
            cur.execute(sql, p)
            return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    rooms = q("""SELECT r.id, r.number, r.sensor_level FROM room r
                 WHERE r.type = 'LECTURE' AND r.sensor_level IN ('FULL', 'SIM') ORDER BY r.id""")
    readings = q("""SELECT room_id, co2, temp, humidity, measured_at FROM sensor_reading
                    WHERE measured_at >= %s AND measured_at <= %s AND room_id = ANY(%s)""",
                 (now - timedelta(days=FIT_DAYS), now, list(map(int, rooms["id"]))))
    readings["measured_at"] = pd.to_datetime(readings["measured_at"])
    sched = q("SELECT id, room_id, day_of_week, start_time, end_time FROM class_schedule")
    cancel = q("SELECT class_schedule_id, cancel_date FROM class_cancellation")
    windows = q("""SELECT room_id, window_opened_at, COALESCE(window_closed_at, completed_at) AS closed_at
                   FROM action WHERE window_opened_at IS NOT NULL AND window_opened_at >= %s""",
                (now - timedelta(days=FIT_DAYS),))
    active = q("""SELECT id, room_id, status, source, recommend, metric, expected_at, window_opened_at, window_closed_at
                  FROM action WHERE status IN ('PENDING', 'CHECKED', 'IN_PROGRESS')""")
    return rooms, readings, Schedule(sched, cancel), windows, active


def run(conn, dry_run=False):
    now = now_kst()
    rooms, readings, schedule, windows, active = load(conn, now)
    pred_rows, new_actions, expire_ids, report = [], [], [], []

    for room in rooms.itertuples():
        rd = readings[readings["room_id"] == room.id]
        if rd.empty or rd["measured_at"].max() < now - timedelta(minutes=10):
            report.append((room.number, "센서 끊김 → 건너뜀"))
            continue
        s = resample(rd)
        latest = rd[rd["measured_at"] >= now - timedelta(minutes=5)]
        c0 = float(latest["co2"].median()) if not latest.empty else float(s["co2"].dropna().iloc[-1])
        temp = s["temp"].dropna().iloc[-1] if s["temp"].notna().any() else None
        hum = s["humidity"].dropna().iloc[-1] if s["humidity"].notna().any() else None

        w = windows[windows["room_id"] == room.id]
        excl = [(r.window_opened_at, r.closed_at if pd.notna(r.closed_at) else now) for r in w.itertuples()]
        params = fit_room(s["co2"].dropna(), room.id, schedule, excl)
        preds = rollout(c0, now, room.id, params, schedule)
        for t, c in preds:
            pred_rows.append((room.id, now, t, int(round(c)),
                              None if temp is None else round(float(temp), 1),
                              None if hum is None else int(round(hum)), MODEL_VERSION))

        # ---- 조치
        mine = active[(active["room_id"] == room.id)]
        window_open = ((mine["status"] == "IN_PROGRESS") & mine["window_opened_at"].notna()
                       & mine["window_closed_at"].isna()).any()
        has_open_action = (mine["recommend"] == "OPEN").any()
        hit = (now, c0) if c0 >= THRESHOLD else first_over(preds, now + timedelta(minutes=ALERT_WINDOW_MIN))

        if hit and not has_open_action and not window_open:
            t_hit, c_hit = hit
            new_actions.append({
                "room_id": room.id, "expected_at": t_hit, "predicted_value": int(round(c_hit)),
                "co2_before": int(round(c0)), "class_schedule_id": schedule.class_at(room.id, t_hit),
                "summary": "창문 열어주세요" if t_hit <= now else f"{int((t_hit - now).total_seconds() // 60)}분 뒤 기준 초과 예상"})
        # 예측 조치인데 이제 넘을 일이 없고, 예상 시각이 30분 넘게 지났으면 만료
        if not hit:
            stale = mine[(mine["source"] == "PREDICTION") & (mine["recommend"] == "OPEN")
                         & (mine["status"].isin(["PENDING", "CHECKED"]))
                         & (pd.to_datetime(mine["expected_at"]) < now - timedelta(minutes=30))]
            expire_ids += list(map(int, stale["id"]))

        max_c = max(c for _, c in preds)
        report.append((room.number, f"지금 {c0:.0f} → 2시간 최고 {max_c:.0f}  "
                                    f"(a={'-' if params['a'] is None else round(params['a'], 1)} b={params['b']:.3f} k={params['k']:.3f} {params['src']}"
                                    f"{' 조치생성' if hit and not has_open_action and not window_open else ''})"))

    if dry_run:
        for n, msg in report:
            print(f"{n:>5}  {msg}")
        print(f"\n예측 {len(pred_rows)}행, 새 조치 {len(new_actions)}건, 만료 {len(expire_ids)}건 (dry-run, 저장 안 함)")
        return

    with conn:
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values
            execute_values(cur, """INSERT INTO prediction (room_id, created_at, target_at, co2, temp, humidity, model_version)
                                   VALUES %s""", pred_rows)
            for a in new_actions:
                cur.execute("""SELECT id FROM prediction WHERE room_id = %s AND created_at = %s AND target_at >= %s
                               ORDER BY target_at LIMIT 1""", (a["room_id"], now, a["expected_at"]))
                pid = cur.fetchone()
                cur.execute("""INSERT INTO action (room_id, source, prediction_id, class_schedule_id, recommend, metric,
                                   predicted_value, threshold, expected_at, summary, co2_before, status, created_at)
                               VALUES (%s, 'PREDICTION', %s, %s, 'OPEN', 'CO2', %s, %s, %s, %s, %s, 'PENDING', %s)""",
                            (a["room_id"], pid[0] if pid else None, a["class_schedule_id"], a["predicted_value"],
                             THRESHOLD, a["expected_at"], a["summary"], a["co2_before"], now))
            if expire_ids:
                cur.execute("UPDATE action SET status = 'EXPIRED' WHERE id = ANY(%s)", (expire_ids,))
            cur.execute("""DELETE FROM prediction p WHERE p.created_at < %s
                           AND NOT EXISTS (SELECT 1 FROM action a WHERE a.prediction_id = p.id)""",
                        (now - timedelta(days=KEEP_PRED_DAYS),))
    print(f"{now:%Y-%m-%d %H:%M} 예측 {len(pred_rows)}행, 새 조치 {len(new_actions)}건, 만료 {len(expire_ids)}건")


# ---------------------------------------------------------------- 백테스트
def backtest(path, room_id=7, sched_path="schedule.csv", cancel_path="cancellations.csv"):
    """과거 각 시점에서, 그 시점 이전 데이터로만 학습 → 30분 뒤 예측 → 실제와 비교 (평일 08~18시)"""
    d = pd.read_csv(path)
    d["measured_at"] = pd.to_datetime(d["measured_at"], format="mixed")
    d = d[d["room_id"].astype(str) == str(room_id)]
    s = resample(d)["co2"].dropna()

    sc = pd.read_csv(sched_path)
    sc["id"] = range(1, len(sc) + 1)
    sc["room_id"] = room_id
    cancel = pd.DataFrame(columns=["class_schedule_id", "cancel_date"])
    if os.path.exists(cancel_path):
        cd = pd.read_csv(cancel_path)
        rows = [(int(r.id), dt) for dt in cd["date"] for r in sc.itertuples()
                if r.day_of_week == pd.Timestamp(dt).dayofweek + 1]
        cancel = pd.DataFrame(rows, columns=["class_schedule_id", "cancel_date"])
    schedule = Schedule(sc, cancel)

    errs_m, errs_p, n_skip = [], [], 0
    params, last_fit = None, None
    for t in s.index:
        if t.weekday() >= 5 or not (8 <= t.hour < 18):
            continue
        target = t + timedelta(minutes=30)
        if target not in s.index or target.hour >= 18:
            continue
        hist = s[(s.index <= t) & (s.index > t - timedelta(days=FIT_DAYS))]
        if len(hist) < 48:
            n_skip += 1
            continue
        if last_fit is None or t - last_fit >= timedelta(hours=1):
            params, last_fit = fit_room(hist, room_id, schedule), t
        pred = rollout(s[t], t, room_id, params, schedule, horizon=30)[-1][1]
        errs_m.append(abs(pred - s[target]))
        errs_p.append(abs(s[t] - s[target]))
    print(f"방 {room_id}, 평가 시점 {len(errs_m)}개 (데이터 부족으로 건너뜀 {n_skip})")
    print(f"그대로 유지(베이스라인) MAE: {np.mean(errs_p):.1f} ppm")
    print(f"예측 모델({MODEL_VERSION}) MAE: {np.mean(errs_m):.1f} ppm")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--backtest":
        backtest(args[1] if len(args) > 1 else "readings.csv", int(args[2]) if len(args) > 2 else 7)
    else:
        c = connect()
        try:
            run(c, dry_run="--dry-run" in args)
        finally:
            c.close()