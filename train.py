"""
617호 CO2 예측 모델 학습 (LightGBM)
- 입력: readings.csv (DB에서 export), schedule.csv (시간표, 직접 작성)
- 출력: 30분 뒤 CO2 예측 모델 + 베이스라인 대비 성능
실행: python train.py
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

STEP_MIN = 5          # 5분 단위로 묶어서 학습
HORIZON_MIN = 30      # 몇 분 뒤 CO2를 예측할지
H = HORIZON_MIN // STEP_MIN
ROOM = "7"            # DB의 room_id (617호)
DAY_START, DAY_END = 8, 18   # 학습에 쓰는 시간대 (야간 수업 있으면 DAY_END 늘리기)


def load_readings(path="readings.csv"):
    df = pd.read_csv(path, parse_dates=["measured_at"], date_format="mixed")
    df = df[df["room_id"].astype(str) == ROOM]
    df = df[df["co2"] < 4900]                  # 센서 오작동값(상한 5000) 제거
    df = (df.set_index("measured_at")[["co2", "temp", "humidity"]]
            .resample(f"{STEP_MIN}min").mean()
            .interpolate(limit=3))          # 15분 이하 공백만 메움
    return df


def load_schedule(path="schedule.csv"):
    s = pd.read_csv(path)                   # day_of_week(1=월~7=일), start_time, end_time, enrolled
    # 수강 인원은 선택: 비어 있으면 0으로 두고 학습에서 뺀다
    s["enrolled"] = pd.to_numeric(s["enrolled"], errors="coerce").fillna(0)
    s["start"] = pd.to_timedelta(s["start_time"] + ":00")
    s["end"] = pd.to_timedelta(s["end_time"] + ":00")
    return s


def load_cancellations(path="cancellations.csv"):
    """휴강한 날짜 목록 (없으면 빈 목록). 형식: date 열에 2026-10-19"""
    try:
        c = pd.read_csv(path)
        return set(pd.to_datetime(c["date"]).dt.date)
    except FileNotFoundError:
        return set()


CANCELLED = set()


def class_info(ts, sched):
    """시각 ts에 수업 중인지, 시작 후 몇 분인지, 다음 수업까지 몇 분인지"""
    dow = ts.dayofweek + 1
    tod = ts - ts.normalize()
    today = sched[sched["day_of_week"] == dow]
    if ts.date() in CANCELLED:                 # 휴강한 날은 수업 없음으로 처리
        today = today.iloc[0:0]

    now = today[(today["start"] <= tod) & (tod < today["end"])]
    in_class = int(len(now) > 0)
    enrolled = int(now["enrolled"].iloc[0]) if in_class else 0
    since_start = (tod - now["start"].iloc[0]).total_seconds() / 60 if in_class else -1

    nxt = today[today["start"] > tod].sort_values("start")
    to_next = (nxt["start"].iloc[0] - tod).total_seconds() / 60 if len(nxt) else 999
    next_enrolled = int(nxt["enrolled"].iloc[0]) if len(nxt) else 0
    return in_class, enrolled, since_start, to_next, next_enrolled


def build_features(df, sched):
    X = pd.DataFrame(index=df.index)
    X["co2"] = df["co2"]
    X["co2_diff_15"] = df["co2"] - df["co2"].shift(3)    # 최근 15분 변화량
    X["temp"] = df["temp"]
    X["humidity"] = df["humidity"]
    X["dow"] = df.index.dayofweek
    X["hour"] = df.index.hour + df.index.minute / 60

    now_info = [class_info(t, sched) for t in df.index]
    fut_info = [class_info(t + pd.Timedelta(minutes=HORIZON_MIN), sched) for t in df.index]
    X[["in_class", "enrolled", "min_since_start", "min_to_next", "next_enrolled"]] = now_info
    X["in_class_at_target"] = [f[0] for f in fut_info]
    X["enrolled_at_target"] = [f[1] for f in fut_info]

    if X["enrolled"].max() == 0 and X["next_enrolled"].max() == 0:
        X = X.drop(columns=["enrolled", "next_enrolled", "enrolled_at_target"])
        print("수강 인원 없음 → 수업 여부·시간 정보로만 학습")

    y = df["co2"].shift(-H)                                # 30분 뒤 실제 CO2
    data = X.assign(target=y).dropna()

    # 수업 없는 시간대 제외: 평일 08:00~18:00만 학습 (피처 계산 '후'에 거르기)
    # 08시부터 남기는 이유: 09시 수업 전 사전 환기 예측이 핵심이라서
    t = data.index
    target_t = t + pd.Timedelta(minutes=HORIZON_MIN)
    keep = ((t.dayofweek < 5) & (t.hour >= DAY_START)
            & (target_t.date == t.date) & (target_t.hour < DAY_END))   # 자정 넘어가는 구간 제외
    data = data[keep]
    return data.drop(columns="target"), data["target"]


def main():
    global CANCELLED
    CANCELLED = load_cancellations()
    if CANCELLED:
        print(f"휴강일 {len(CANCELLED)}일 반영: {sorted(CANCELLED)}")
    df = load_readings()
    sched = load_schedule()
    X, y = build_features(df, sched)
    print(f"학습 데이터: {len(X)}행 ({X.index.min()} ~ {X.index.max()})")

    split = int(len(X) * 0.8)                              # 시간 순서대로 앞 80% 학습, 뒤 20% 평가
    X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              min_child_samples=10, verbose=-1)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    mae_model = mean_absolute_error(y_te, pred)
    mae_base = mean_absolute_error(y_te, X_te["co2"])     # 베이스라인: "지금 값이 30분 뒤에도 그대로"
    print(f"베이스라인 MAE: {mae_base:.0f} ppm")
    print(f"모델 MAE:       {mae_model:.0f} ppm  ({(1 - mae_model / mae_base) * 100:.0f}% 개선)")

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n피처 중요도:\n", imp.to_string())

    model.booster_.save_model("co2_model.txt")
    print("\n모델 저장: co2_model.txt")


if __name__ == "__main__":
    main()
