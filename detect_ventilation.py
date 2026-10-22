"""
617호 센서 데이터만으로 환기(창문 열림) 구간을 자동 감지하고, 환기 효과를 학습한다.

원리: CO2는 바깥 농도(약 420ppm)로 지수적으로 수렴한다.
      dC/dt = -k (C - C_out)
      k(시간당 환기 횟수)가 문 닫혔을 때보다 몇 배 크면 → 창문이 열린 것.

실행: python detect_ventilation.py [readings.csv]
출력: ventilation_events.csv (감지된 환기 구간) + 환기 효과 모델 요약
"""
import sys
import numpy as np
import pandas as pd

ROOM = "7"            # DB의 room_id (617호)
MIN_EXCESS = 250      # 바깥보다 250ppm 이상 높을 때만 판단 (낮으면 잡음에 묻힘)
RATIO = 3.0           # 평소보다 k가 3배 이상이면 환기로 판단
MIN_K = 2.0           # 단, 시간당 2회 이상은 되어야 함
MIN_MINUTES = 4       # 4분 이상 이어져야 환기로 인정 (문 잠깐 여닫힘 제외)


def load(path):
    df = pd.read_csv(path, parse_dates=["measured_at"], date_format="mixed")
    df = df[df["room_id"].astype(str) == ROOM]
    # 센서 이상값 제거: MH-Z19 상한(5000)에 붙은 값, 1분 만에 1000ppm 넘게 튄 값
    df = df[df["co2"] < 4900]
    jump = df["co2"].diff().abs()
    df = df[~((jump > 1000) & (df["co2"].diff(-1).abs() > 1000))]
    s = (df.set_index("measured_at")["co2"]
           .resample("1min").mean()
           .interpolate(limit=3))
    return s.rolling(3, center=True, min_periods=1).median()   # 측정 잡음 제거


def decay_rate(co2, c_out):
    """분 단위 CO2로 시간당 감쇠율 k 계산 (3분 기울기)"""
    slope = (co2.shift(-1) - co2.shift(1)) / 2          # ppm/분
    excess = (co2 - c_out).clip(lower=1)
    return -slope / excess * 60                         # 1/시간


def find_events(co2):
    c_out = co2.quantile(0.02)                          # 새벽 최저치 ≈ 바깥 농도
    k = decay_rate(co2, c_out)
    valid = (co2 - c_out) > MIN_EXCESS
    k_closed = k[valid & (k > 0)].median()              # 평소(문 닫힘) 감쇠율
    thr = max(RATIO * k_closed, MIN_K)

    is_vent = valid & (k > thr)
    groups = (is_vent != is_vent.shift()).cumsum()

    events = []
    for _, g in is_vent.groupby(groups):
        if not g.iloc[0] or len(g) < MIN_MINUTES:
            continue
        start, end = g.index[0] - pd.Timedelta(minutes=1), g.index[-1]
        c0, c1 = co2[start], co2[end]
        minutes = (end - start).total_seconds() / 60
        k_open = np.log((c0 - c_out) / max(c1 - c_out, 1)) / minutes * 60
        events.append({"start": start, "end": end, "minutes": round(minutes),
                       "co2_start": round(c0), "co2_end": round(c1),
                       "k_per_hour": round(k_open, 1)})
    return pd.DataFrame(events), c_out, k_closed, thr


def minutes_needed(c_now, target, c_out, k_open):
    """지금 CO2에서 목표치까지 창문을 몇 분 열어야 하는지"""
    if c_now <= target:
        return 0
    return int(np.ceil(np.log((c_now - c_out) / (target - c_out)) / k_open * 60))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "readings.csv"
    co2 = load(path)
    ev, c_out, k_closed, thr = find_events(co2)

    print(f"바깥 기준 CO2: {c_out:.0f} ppm")
    print(f"문 닫힘 감쇠율: 시간당 {k_closed:.2f}회 → 환기 판단 기준: {thr:.1f}회 이상")
    print(f"감지된 환기 구간: {len(ev)}개\n")
    if ev.empty:
        print("환기 구간이 없어요. 데이터가 더 쌓이면 다시 돌려보세요.")
        return
    print(ev.to_string(index=False))
    ev.to_csv("ventilation_events.csv", index=False)

    k_open = ev["k_per_hour"].median()
    print(f"\n창문 열었을 때 감쇠율(중앙값): 시간당 {k_open:.1f}회 "
          f"(문 닫힘의 {k_open / k_closed:.0f}배)")
    for c_now in (1200, 1500, 2000):
        print(f"  CO2 {c_now} → 800 까지: 창문 {minutes_needed(c_now, 800, c_out, k_open)}분")


if __name__ == "__main__":
    main()
