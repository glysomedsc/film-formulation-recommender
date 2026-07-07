# -*- coding: utf-8 -*-
"""
추천 엔진 — 사용자가 준 물성 제약을 만족하는 최적 배합을 계산.

핵심:
  - 오라클(HistGBM 회귀모델, R2>=0.9)이 조성/공정 -> 물성을 예측.
  - 사용자는 '최대화할 물성 1개' + '나머지 물성의 하한(>=)'을 준다.
  - 제약을 만족하는 배합 공간에서 목표 물성을 최대화한다.
    (오라클이 매우 빠르므로: 조밀 랜덤 샘플링으로 전역 탐색 -> 국소 정밀화)
  - 조성 합=100 제약은 compatibilizer를 잔여량으로 두어 항상 보장.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

FEATURES = ["resin_A_ratio", "resin_B_ratio", "plasticizer", "compatibilizer",
            "uv_stabilizer", "process_temp", "draw_ratio"]
TARGETS  = ["tensile_strength", "transparency", "barrier"]
FREE     = ["resin_A_ratio", "resin_B_ratio", "plasticizer",
            "uv_stabilizer", "process_temp", "draw_ratio"]
UNITS = {
    "resin_A_ratio": "%", "resin_B_ratio": "%", "plasticizer": "%",
    "compatibilizer": "%", "uv_stabilizer": "phr", "process_temp": "℃",
    "draw_ratio": "",
}


class Recommender:
    def __init__(self, csv_path="film_experiments.csv"):
        self.df = pd.read_csv(csv_path)
        self.lo = self.df[FREE].min().values.astype(float)
        self.hi = self.df[FREE].max().values.astype(float)
        self.compat_lo = float(self.df["compatibilizer"].min())
        self.compat_hi = float(self.df["compatibilizer"].max())
        self.y_min = self.df[TARGETS].min().values.astype(float)
        self.y_max = self.df[TARGETS].max().values.astype(float)
        self.y_rng = np.where(self.y_max - self.y_min == 0, 1.0, self.y_max - self.y_min)
        self.oracle, self.r2 = self._train_oracle()

    # ---------- 오라클 학습 ----------
    def _train_oracle(self):
        X = self.df[FEATURES].values
        y = self.df[TARGETS].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        m = MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, random_state=42))
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        r2 = {t: float(r2_score(yte[:, i], yp[:, i])) for i, t in enumerate(TARGETS)}
        # 오라클은 전체 데이터로 재학습
        m.fit(X, y)
        return m, r2

    # ---------- 조성 제약(합=100) 처리 ----------
    def _to_full(self, Xfree):
        """자유변수(n,6) -> 전체(n,7). compatibilizer=100-조성3."""
        a, b, p = Xfree[:, 0], Xfree[:, 1], Xfree[:, 2]
        compat = 100.0 - a - b - p
        return np.column_stack([a, b, p, compat, Xfree[:, 3], Xfree[:, 4], Xfree[:, 5]])

    def _sample_feasible(self, n, rng, lo=None, hi=None):
        """compatibilizer가 유효범위 안에 드는 배합 n개를 벡터화 생성."""
        lo = self.lo if lo is None else lo
        hi = self.hi if hi is None else hi
        kept = []
        got = 0
        while got < n:
            m = max(int((n - got) * 6), 256)
            U = rng.uniform(lo, hi, size=(m, len(FREE)))
            compat = 100.0 - U[:, 0] - U[:, 1] - U[:, 2]
            ok = (compat >= self.compat_lo) & (compat <= self.compat_hi)
            if ok.any():
                kept.append(U[ok]); got += int(ok.sum())
        return np.vstack(kept)[:n]

    # ---------- 추천 ----------
    def _score(self, preds, maximize):
        """목표 점수 벡터. maximize 지정시 해당 물성, 없으면 세 물성 정규화 평균(균형)."""
        if maximize is not None:
            return preds[:, TARGETS.index(maximize)]
        norm = (preds - self.y_min) / self.y_rng
        return norm.mean(axis=1)

    def recommend(self, maximize=None, mins=None, n_global=40000, n_local=8000, seed=0):
        """
        maximize: 최대화할 타깃명 (TARGETS 중 하나) 또는 None(균형 최적)
        mins: {타깃명: 하한값} — 만족해야 할 제약 (없으면 {})
        반환: dict(formulation, predicted, feasible, satisfied, ...)
        """
        if maximize is not None:
            assert maximize in TARGETS
        mins = {k: float(v) for k, v in (mins or {}).items() if v is not None}
        rng = np.random.default_rng(seed)

        def eval_pool(Xfree):
            full = self._to_full(Xfree)
            preds = self.oracle.predict(full)
            mask = np.ones(len(Xfree), bool)
            for tgt, lo in mins.items():
                mask &= preds[:, TARGETS.index(tgt)] >= lo
            return full, preds, mask

        # 1) 전역 조밀 탐색
        Xf = self._sample_feasible(n_global, rng)
        full, preds, mask = eval_pool(Xf)

        feasible = bool(mask.any())
        score = self._score(preds, maximize)
        if feasible:
            pool = np.where(mask)[0]
            best = pool[int(np.argmax(score[pool]))]
        else:
            # 제약 만족 배합이 없으면: 정규화 위반량 최소인 배합을 best-effort로
            viol = np.zeros(len(Xf))
            for tgt, lo in mins.items():
                j = TARGETS.index(tgt)
                viol += np.maximum(0.0, (lo - preds[:, j]) / self.y_rng[j])
            best = int(np.argmin(viol))

        # 2) 국소 정밀화 — best 주변 축소 상자에서 재탐색 (제약 만족시에만 목표 개선 시도)
        if feasible and n_local > 0:
            center = Xf[best]
            width = 0.12 * (self.hi - self.lo)
            lo_b = np.maximum(self.lo, center - width)
            hi_b = np.minimum(self.hi, center + width)
            Xf2 = self._sample_feasible(n_local, rng, lo_b, hi_b)
            full2, preds2, mask2 = eval_pool(Xf2)
            if mask2.any():
                score2 = self._score(preds2, maximize)
                pool2 = np.where(mask2)[0]
                cand = pool2[int(np.argmax(score2[pool2]))]
                if score2[cand] > score[best]:
                    full, preds, best = full2, preds2, cand

        formulation = {name: float(v) for name, v in zip(FEATURES, full[best])}
        predicted   = {t: float(v) for t, v in zip(TARGETS, preds[best])}
        satisfied   = {t: bool(predicted[t] >= lo) for t, lo in mins.items()}

        return {
            "maximize": maximize,
            "constraints": mins,
            "feasible": feasible,
            "satisfied": satisfied,
            "formulation": formulation,
            "predicted": predicted,
            "units": UNITS,
            "oracle_r2": self.r2,
        }


if __name__ == "__main__":
    rec = Recommender()
    print("오라클 R2:", {k: round(v, 3) for k, v in rec.r2.items()})
    out = rec.recommend(maximize="barrier",
                        mins={"transparency": 85, "tensile_strength": 50})
    print("feasible:", out["feasible"], "satisfied:", out["satisfied"])
    print("추천 배합:")
    for k, v in out["formulation"].items():
        print(f"  {k:16s} {v:8.2f} {UNITS[k]}")
    print("예상 물성:")
    for k, v in out["predicted"].items():
        print(f"  {k:16s} {v:8.2f}")
