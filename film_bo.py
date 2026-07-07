# -*- coding: utf-8 -*-
"""
필름 배합 최적화 — 최소 실험으로 물성 최대화
================================================
과제:
  1) 조성·공정(7) -> 물성(3) 회귀 모델을 만들어 '실험을 대신하는 오라클'로 사용 (R2 >= 0.9)
  2) 그 위에서 베이지안 최적화로 최적 배합을 '최소 실험'으로 탐색
  3) 랜덤 탐색 대비 적은 실험 횟수로 더 높은 물성에 도달함을 수렴 곡선으로 입증

핵심 아이디어:
  - "실험 1회" = "오라클(회귀모델) 1회 쿼리". 실험 비용 = 쿼리 횟수.
  - BO는 관측을 GP로 모델링하고 획득함수(EI)로 '다음 실험점'을 고른다.
    -> 유망한 곳(exploitation)과 안 가본 곳(exploration)의 균형은 EI의 xi로 제어.
  - BO vs 랜덤 탐색을 같은 예산/같은 오라클/여러 시드 평균으로 공정 비교.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

RNG_GLOBAL = 42

FEATURES = ["resin_A_ratio", "resin_B_ratio", "plasticizer", "compatibilizer",
            "uv_stabilizer", "process_temp", "draw_ratio"]
TARGETS  = ["tensile_strength", "transparency", "barrier"]

# 탐색 자유변수 6개 (compatibilizer는 잔여량으로 종속 -> 조성 합=100 보장)
FREE = ["resin_A_ratio", "resin_B_ratio", "plasticizer",
        "uv_stabilizer", "process_temp", "draw_ratio"]


# =====================================================================
# 1. 데이터 & 오라클(회귀모델) 학습 + R2 검증
# =====================================================================
def load_data():
    df = pd.read_csv("film_experiments.csv")
    return df


def build_oracle(df):
    """조성/공정 -> 물성 회귀모델. R2>=0.9 검증 후, 전 데이터로 재학습해 오라클로 반환."""
    X = df[FEATURES].values
    y = df[TARGETS].values

    candidates = {
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            random_state=RNG_GLOBAL, n_jobs=-1)),
        "ExtraTrees": MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=1,
            random_state=RNG_GLOBAL, n_jobs=-1)),
        "HistGBM": MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=None,
            random_state=RNG_GLOBAL)),
    }

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RNG_GLOBAL)

    print("=" * 68)
    print("1) 오라클 후보 비교  (test_size=0.2, held-out R2)")
    print("=" * 68)
    best_name, best_model, best_mean = None, None, -1e9
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        r2s = [r2_score(yte[:, i], yp[:, i]) for i in range(len(TARGETS))]
        mean_r2 = float(np.mean(r2s))
        tag = "  <== 선택" if mean_r2 > best_mean else ""
        print(f"  {name:14s}  R2: " +
              "  ".join(f"{t}={r:.3f}" for t, r in zip(TARGETS, r2s)) +
              f"   |평균={mean_r2:.3f}{tag}")
        if mean_r2 > best_mean:
            best_name, best_model, best_mean = name, model, mean_r2

    # 선택 모델 상세 리포트
    best_model.fit(Xtr, ytr)
    yp = best_model.predict(Xte)
    print("-" * 68)
    print(f"선택 오라클: {best_name}")
    all_ok = True
    for i, t in enumerate(TARGETS):
        r2 = r2_score(yte[:, i], yp[:, i]); mae = mean_absolute_error(yte[:, i], yp[:, i])
        ok = r2 >= 0.9
        all_ok &= ok
        print(f"  {t:18s}  R2={r2:.3f}  MAE={mae:.2f}   목표(>=0.9) {'달성' if ok else '미달'}")
    print(f"  => 3개 타깃 모두 R2>=0.9: {'예' if all_ok else '아니오'}")

    # 오라클은 전체 데이터로 재학습 (실험을 대신하는 '진짜 시스템')
    oracle = candidates[best_name]
    oracle.fit(X, y)
    return oracle, best_name, all_ok


# =====================================================================
# 2. 목적함수 — 물성 3개(트레이드오프)를 스칼라로
# =====================================================================
class Objective:
    """정규화 가중합. 기본은 세 물성 동등 가중(=전부 최대화)."""
    def __init__(self, df, weights=None):
        self.y_min = df[TARGETS].min().values.astype(float)
        self.y_max = df[TARGETS].max().values.astype(float)
        self.y_rng = np.where(self.y_max - self.y_min == 0, 1.0, self.y_max - self.y_min)
        w = np.ones(len(TARGETS)) if weights is None else np.asarray(weights, float)
        self.w = w / w.sum()

    def scalar(self, props):
        norm = (np.asarray(props) - self.y_min) / self.y_rng
        return float(np.dot(norm, self.w))


# =====================================================================
# 3. 탐색 공간 & 오라클 쿼리 (실험 1회)
# =====================================================================
class SearchSpace:
    """자유변수 6개의 [min,max] 경계. compatibilizer는 잔여량(제약)."""
    def __init__(self, df):
        self.lo = df[FREE].min().values.astype(float)
        self.hi = df[FREE].max().values.astype(float)
        self.compat_lo = float(df["compatibilizer"].min())
        self.compat_hi = float(df["compatibilizer"].max())

    def to_full(self, xfree):
        """자유변수 -> 전체 7변수 (compatibilizer=100-조성3). 유효하면 벡터, 아니면 None."""
        rA, rB, plast, uv, temp, draw = xfree
        compat = 100.0 - rA - rB - plast
        if compat < self.compat_lo or compat > self.compat_hi:
            return None  # 조성 제약 위반 -> 실현 불가능한 배합
        return np.array([rA, rB, plast, compat, uv, temp, draw])

    def sample_feasible(self, n, rng):
        """제약을 만족하는 임의 배합 n개 생성 (rejection sampling)."""
        out = []
        while len(out) < n:
            batch = rng.uniform(self.lo, self.hi, size=(max(n * 4, 64), len(FREE)))
            for xf in batch:
                full = self.to_full(xf)
                if full is not None:
                    out.append(xf)
                    if len(out) >= n:
                        break
        return np.array(out[:n])


def make_experiment(oracle, space, obj):
    """실험(=오라클 쿼리): 자유변수 배합 -> 스칼라 물성 점수. 실현불가면 매우 낮은 값."""
    def run(xfree):
        full = space.to_full(xfree)
        if full is None:
            return -1.0  # 실현 불가능한 배합에 패널티
        props = oracle.predict(full.reshape(1, -1))[0]
        return obj.scalar(props)
    return run


# =====================================================================
# 4. 베이지안 최적화 (GP 대리모델 + EI 획득함수)
# =====================================================================
def expected_improvement(mu, sigma, f_best, xi):
    """최대화용 EI. xi가 클수록 exploration(안 가본 곳)을 더 선호."""
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - f_best - xi
    Z = imp / sigma
    return imp * norm.cdf(Z) + sigma * norm.pdf(Z)


def bayes_opt(experiment, space, budget, seed, n_init=8, xi=0.01, pool=3000):
    """
    budget번 실험으로 스칼라 점수 최대화.
    - 초기 n_init개: 랜덤(공간 파악)
    - 이후: GP를 관측에 적합 -> 후보풀에서 EI 최대점을 '다음 실험'으로 선택
    반환: best_so_far[budget]  (실험 i회까지의 최고 점수)
    """
    rng = np.random.default_rng(seed)
    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=np.ones(len(FREE)), nu=2.5)
              + WhiteKernel(1e-3, (1e-6, 1e0)))

    span = (space.hi - space.lo)
    scale = lambda X: (X - space.lo) / span  # GP 입력 정규화

    X_obs = space.sample_feasible(n_init, rng)
    y_obs = np.array([experiment(x) for x in X_obs])

    best_curve = list(np.maximum.accumulate(y_obs))

    for _ in range(n_init, budget):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=seed)
        gp.fit(scale(X_obs), y_obs)

        cand = space.sample_feasible(pool, rng)
        mu, sigma = gp.predict(scale(cand), return_std=True)
        ei = expected_improvement(mu, sigma, f_best=y_obs.max(), xi=xi)
        x_next = cand[int(np.argmax(ei))]

        y_next = experiment(x_next)
        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)
        best_curve.append(max(best_curve[-1], y_next))

    return np.array(best_curve), X_obs, y_obs


def random_search(experiment, space, budget, seed):
    """랜덤 탐색 베이스라인: 매번 임의 배합을 실험."""
    rng = np.random.default_rng(seed)
    X = space.sample_feasible(budget, rng)
    y = np.array([experiment(x) for x in X])
    return np.maximum.accumulate(y), X, y


# =====================================================================
# 5. 실행 — BO vs 랜덤, 여러 시드 평균 + 수렴 곡선
# =====================================================================
def main():
    df = load_data()
    oracle, oracle_name, r2_ok = build_oracle(df)

    obj = Objective(df)                 # 동등 가중(세 물성 모두 최대화)
    space = SearchSpace(df)
    experiment = make_experiment(oracle, space, obj)

    # 기존 600개 실험 중 최고 점수 (베이스라인 기준선)
    base_scores = np.array([obj.scalar(row) for row in df[TARGETS].values])
    base_best = base_scores.max()

    BUDGET = 40
    SEEDS = list(range(15))

    print("\n" + "=" * 68)
    print(f"2) 최적 배합 탐색  |  예산={BUDGET}회 실험  |  시드 {len(SEEDS)}개 평균")
    print("=" * 68)

    bo_curves, rand_curves = [], []
    best_overall = (-1e9, None)
    for s in SEEDS:
        bo_c, bo_X, bo_y = bayes_opt(experiment, space, BUDGET, seed=s)
        rd_c, _, _ = random_search(experiment, space, BUDGET, seed=1000 + s)
        bo_curves.append(bo_c); rand_curves.append(rd_c)
        i = int(np.argmax(bo_y))
        if bo_y[i] > best_overall[0]:
            best_overall = (bo_y[i], bo_X[i])

    bo_curves = np.array(bo_curves); rand_curves = np.array(rand_curves)
    bo_mean, bo_std = bo_curves.mean(0), bo_curves.std(0)
    rd_mean, rd_std = rand_curves.mean(0), rand_curves.std(0)
    xs = np.arange(1, BUDGET + 1)

    # --- 표본 효율 지표: 랜덤이 40회에 도달한 최고치를 BO는 몇 회에 도달? ---
    rand_final = rd_mean[-1]
    reach = np.argmax(bo_mean >= rand_final)
    reach = reach + 1 if bo_mean[-1] >= rand_final else None
    speedup = (BUDGET / reach) if reach else None

    print(f"  랜덤 탐색 최종 평균 점수 (40회): {rand_final:.4f}")
    print(f"  BO       최종 평균 점수 (40회): {bo_mean[-1]:.4f}")
    if reach:
        print(f"  => BO는 {reach}회 실험만에 랜덤의 40회 성과를 따라잡음 "
              f"(약 {speedup:.1f}배 적은 실험)")
    print(f"  기존 600개 실험 중 최고: {base_best:.4f}  "
          f"| BO 발견 최고: {best_overall[0]:.4f}")

    # --- 최적 배합 리포트 ---
    full = space.to_full(best_overall[1])
    props = oracle.predict(full.reshape(1, -1))[0]
    print("\n" + "-" * 68)
    print("BO가 찾은 최적 배합 (오라클 예측):")
    for name, v in zip(FEATURES, full):
        print(f"  {name:18s}: {v:8.3f}")
    print("  " + "-" * 30)
    for t, v in zip(TARGETS, props):
        print(f"  {t:18s}: {v:8.3f}")
    print(f"  스칼라 점수: {obj.scalar(props):.4f}")

    # =================================================================
    # 6. 수렴 곡선
    # =================================================================
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(xs, bo_mean, color="#c0392b", lw=2.3, label="Bayesian Optimization (EI)")
    ax.fill_between(xs, bo_mean - bo_std, bo_mean + bo_std, color="#c0392b", alpha=0.15)
    ax.plot(xs, rd_mean, color="#2c3e50", lw=2.3, ls="--", label="Random Search")
    ax.fill_between(xs, rd_mean - rd_std, rd_mean + rd_std, color="#2c3e50", alpha=0.12)
    ax.axhline(base_best, color="#7f8c8d", lw=1.2, ls=":",
               label=f"Best of 600 existing exps ({base_best:.3f})")
    if reach:
        ax.axvline(reach, color="#27ae60", lw=1.2, ls="-.", alpha=0.8)
        ax.annotate(f"BO matches Random@40\nin {reach} experiments",
                    xy=(reach, rand_final), xytext=(reach + 3, rand_final - 0.06),
                    fontsize=9, color="#27ae60",
                    arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax.set_xlabel("Number of experiments (oracle queries)")
    ax.set_ylabel("Best property score found so far  (normalized weighted sum)")
    ax.set_title(f"Sample Efficiency: BO vs Random  (oracle={oracle_name}, mean of {len(SEEDS)} seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("bo_vs_random.png", dpi=130)
    print("\n수렴 곡선 저장: bo_vs_random.png")


if __name__ == "__main__":
    main()
