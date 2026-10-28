"""
예측 모델 성능 그래프 (matplotlib)

입력
  readings.csv       : DB에서 뽑은 측정값 (room_id, co2, measured_at 열 필요)
  schedule_all.csv   : 강의실별 시간표 (room, day_of_week, start_time, end_time)
  cancellations.csv  : 617호 휴강일 (date)

출력 (같은 폴더에 PNG 저장)
  eval_mae.png       : 방별 30분 뒤 예측 오차(MAE) — 예측 모델 vs 지금 값 그대로
  eval_timeline.png  : 한 방의 하루 수업 — 실제 vs 30분 전에 한 예측
  eval_scatter.png   : 실제 vs 예측 산점도 (대각선에 가까울수록 정확)

실행
  pip install matplotlib
  python plot_eval.py                          # 기본: 508호 10/26 타임라인
  python plot_eval.py --room 508 --day 2026-10-26
"""
import argparse
import sys
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import predict as P

# 한글 폰트 (윈도우: 맑은 고딕, 맥: AppleGothic, 리눅스: 나눔고딕)
for font in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False

ROOM_IDS = {"617": 7, "618": 6, "621": 1, "620": 2, "608": 13,
            "508": 18, "517": 19, "518": 20, "520": 21, "521": 22}
REAL_ROOMS = {"617"}            # 실측 센서 방 (나머지는 시뮬레이션)
HORIZON = 30


def load(readings_path, sched_path, cancel_path):
    d = pd.read_csv(readings_path)
    d["measured_at"] = pd.to_datetime(d["measured_at"], format="mixed")
    for col in ("temp", "humidity"):          # 없으면 빈 열로 (CO2만 평가)
        if col not in d.columns:
            d[col] = np.nan
    sa = pd.read_csv(sched_path)
    try:
        cancel_dates = list(pd.read_csv(cancel_path)["date"])
    except FileNotFoundError:
        cancel_dates = []
    return d, sa, cancel_dates


def schedule_for(sa, number, room_id, cancel_dates):
    sc = sa[sa["room"].astype(str) == number].copy()
    sc["id"] = range(1, len(sc) + 1)
    sc["room_id"] = room_id
    canc = []
    if number in REAL_ROOMS:          # 휴강 기록은 617호 것만 있음
        canc = [(int(r.id), dt) for dt in cancel_dates for r in sc.itertuples()
                if r.day_of_week == pd.Timestamp(dt).dayofweek + 1]
    return P.Schedule(sc, pd.DataFrame(canc, columns=["class_schedule_id", "cancel_date"]))


def backtest(series, room_id, sch, start=None):
    """각 시점에서 그 이전 7일로만 학습 → 30분 뒤 예측. 평일 08~18시만 평가"""
    rows, params, last = [], None, None
    excl = P.manual_exclude(room_id)
    in_excl = lambda x: any(a <= x <= b for a, b in excl)
    for ts in series.index:
        if start is not None and ts < start:
            continue
        if ts.weekday() >= 5 or not (8 <= ts.hour < 18):
            continue
        tg = ts + timedelta(minutes=HORIZON)
        if tg not in series.index or tg.hour >= 18:
            continue
        if in_excl(ts) or in_excl(tg):           # 창문 열어둔 기간은 평가에서 제외
            continue
        hist = series[(series.index <= ts) & (series.index > ts - timedelta(days=P.FIT_DAYS))]
        if len(hist) < 48:
            continue
        if last is None or ts - last >= timedelta(hours=1):
            params, last = P.fit_room(hist, room_id, sch, excl), ts
        pred = P.rollout(series[ts], ts, room_id, params, sch, horizon=HORIZON)[-1][1]
        rows.append((ts, tg, series[ts], series[tg], pred, sch.class_at(room_id, tg) is not None))
    return pd.DataFrame(rows, columns=["t", "target", "now", "true", "pred", "in_class"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", default="readings.csv")
    ap.add_argument("--schedule", default="schedule_all.csv")
    ap.add_argument("--cancel", default="cancellations.csv")
    ap.add_argument("--room", default="508", help="타임라인 그래프에 쓸 호실")
    ap.add_argument("--day", default="2026-10-26", help="타임라인 그래프 날짜")
    args = ap.parse_args()

    d, sa, cancel_dates = load(args.readings, args.schedule, args.cancel)
    results, all_bt = [], {}
    for number, rid in ROOM_IDS.items():
        rd = d[d["room_id"] == rid]
        if rd.empty:
            continue
        s = P.resample(rd)["co2"].dropna()
        sch = schedule_for(sa, number, rid, cancel_dates)
        bt = backtest(s, rid, sch)
        if bt.empty:
            continue
        all_bt[number] = bt
        c = bt[bt["in_class"]]
        results.append({
            "room": number, "real": number in REAL_ROOMS,
            "mae_model": (bt["pred"] - bt["true"]).abs().mean(),
            "mae_base": (bt["now"] - bt["true"]).abs().mean(),
            "mae_model_class": (c["pred"] - c["true"]).abs().mean() if len(c) else np.nan,
            "mae_base_class": (c["now"] - c["true"]).abs().mean() if len(c) else np.nan,
        })
    res = pd.DataFrame(results)
    if res.empty:
        sys.exit("평가할 데이터가 없어요. readings.csv 를 확인하세요.")
    print(res.round(1).to_string(index=False))

    # 1) 방별 MAE 막대그래프 (수업 중)
    labels = [f"{r.room}\n({'실측' if r.real else '시뮬'})" for r in res.itertuples()]
    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, res["mae_model_class"], 0.4, label="예측 모델", color="#4C8BF5")
    ax.bar(x + 0.2, res["mae_base_class"], 0.4, label="지금 값 그대로", color="#C9CED6")
    for i, (m, b) in enumerate(zip(res["mae_model_class"], res["mae_base_class"])):
        if not np.isnan(m):
            ax.text(i - 0.2, m + 4, f"{m:.0f}", ha="center", fontsize=9)
            ax.text(i + 0.2, b + 4, f"{b:.0f}", ha="center", fontsize=9, color="#666")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE (ppm)")
    ax.set_title(f"수업 중 {HORIZON}분 뒤 CO2 예측 오차 (낮을수록 정확)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("eval_mae.png", dpi=150)

    # 2) 하루 타임라인
    if args.room in all_bt:
        bt = all_bt[args.room]
        day = bt[bt["target"].dt.date == pd.Timestamp(args.day).date()]
        if not day.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(day["target"], day["true"], color="#222", lw=2, label="실제")
            ax.plot(day["target"], day["pred"], color="#4C8BF5", lw=2, ls="--", label=f"예측 모델 ({HORIZON}분 전)")
            ax.plot(day["target"], day["now"], color="#AAB2BD", lw=1.5, ls=":", label=f"지금 값 그대로 ({HORIZON}분 전 값)")
            ax.axhline(P.THRESHOLD, color="#E5484D", lw=1, ls="--")
            ax.text(day["target"].iloc[0], P.THRESHOLD + 15, f"기준 {P.THRESHOLD}ppm", color="#E5484D", fontsize=9)
            kind = "실측" if args.room in REAL_ROOMS else "시뮬레이션"
            ax.set_title(f"{args.room}호 {args.day} — {HORIZON}분 전에 한 예측 vs 실제 ({kind})")
            ax.set_ylabel("CO2 (ppm)")
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
            ax.legend(frameon=False, loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig("eval_timeline.png", dpi=150)
        else:
            print(f"{args.room}호 {args.day} 데이터가 없어서 타임라인은 건너뜀")

    # 3) 실제 vs 예측 산점도 (전체 방, 수업 중)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for number, bt in all_bt.items():
        c = bt[bt["in_class"]]
        if c.empty:
            continue
        ax.scatter(c["true"], c["pred"], s=10, alpha=0.5,
                   label=f"{number} ({'실측' if number in REAL_ROOMS else '시뮬'})")
    lo, hi = 350, max(1700, ax.get_xlim()[1])
    ax.plot([lo, hi], [lo, hi], color="#999", lw=1, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("실제 CO2 (ppm)"); ax.set_ylabel(f"{HORIZON}분 전 예측 (ppm)")
    ax.set_title("실제 vs 예측 (수업 중, 대각선에 가까울수록 정확)")
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("eval_scatter.png", dpi=150)

    print("\n저장: eval_mae.png, eval_timeline.png, eval_scatter.png")


if __name__ == "__main__":
    main()
