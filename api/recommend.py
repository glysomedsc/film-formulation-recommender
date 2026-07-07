# -*- coding: utf-8 -*-
"""
Vercel 서버리스 함수 — POST /api/recommend
조성/공정 -> 물성 오라클(HistGBM)로, 제약(하한>=)을 만족하며 목표 물성을
최대화(또는 균형)하는 배합을 추천. 콜드스타트 시 1회 학습 후 모듈 전역에 캐시.
pandas/scipy 직접 의존 없이 numpy + scikit-learn 만 사용(함수 용량 절감).
"""
import os
import json
from http.server import BaseHTTPRequestHandler

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

FEATURES = ["resin_A_ratio", "resin_B_ratio", "plasticizer", "compatibilizer",
            "uv_stabilizer", "process_temp", "draw_ratio"]
TARGETS  = ["tensile_strength", "transparency", "barrier"]
FREE_IDX = [0, 1, 2, 4, 5, 6]      # 자유변수(조성3 + uv/temp/draw), compatibilizer(3)는 잔여량
UNITS = {"resin_A_ratio": "%", "resin_B_ratio": "%", "plasticizer": "%",
         "compatibilizer": "%", "uv_stabilizer": "phr", "process_temp": "℃", "draw_ratio": ""}

_STATE = None


def _find_csv():
    here = os.path.dirname(__file__)
    for p in (os.path.join(here, "..", "film_experiments.csv"),
              os.path.join(os.getcwd(), "film_experiments.csv"),
              "film_experiments.csv",
              os.path.join(here, "film_experiments.csv")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("film_experiments.csv not found")


def _state():
    """오라클 + 탐색범위. 콜드스타트당 1회 학습."""
    global _STATE
    if _STATE is not None:
        return _STATE
    data = np.genfromtxt(_find_csv(), delimiter=",", skip_header=1)
    X, Y = data[:, :7], data[:, 7:10]
    lo, hi = X[:, FREE_IDX].min(0), X[:, FREE_IDX].max(0)
    compat_lo, compat_hi = float(X[:, 3].min()), float(X[:, 3].max())
    y_min, y_max = Y.min(0), Y.max(0)
    y_rng = np.where(y_max - y_min == 0, 1.0, y_max - y_min)

    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    m = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, random_state=42))
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    r2 = {TARGETS[i]: round(float(r2_score(yte[:, i], yp[:, i])), 4) for i in range(3)}
    m.fit(X, Y)  # 전체 데이터로 재학습(오라클)

    _STATE = dict(oracle=m, lo=lo, hi=hi, compat_lo=compat_lo, compat_hi=compat_hi,
                  y_min=y_min, y_max=y_max, y_rng=y_rng, r2=r2)
    return _STATE


def _to_full(Xf):
    a, b, p = Xf[:, 0], Xf[:, 1], Xf[:, 2]
    compat = 100.0 - a - b - p
    return np.column_stack([a, b, p, compat, Xf[:, 3], Xf[:, 4], Xf[:, 5]])


def _sample_feasible(st, n, rng, lo=None, hi=None):
    lo = st["lo"] if lo is None else lo
    hi = st["hi"] if hi is None else hi
    kept, got = [], 0
    while got < n:
        U = rng.uniform(lo, hi, size=(max((n - got) * 6, 256), 6))
        compat = 100.0 - U[:, 0] - U[:, 1] - U[:, 2]
        ok = (compat >= st["compat_lo"]) & (compat <= st["compat_hi"])
        if ok.any():
            kept.append(U[ok]); got += int(ok.sum())
    return np.vstack(kept)[:n]


def _score(st, preds, maximize):
    if maximize is not None:
        return preds[:, TARGETS.index(maximize)]
    return ((preds - st["y_min"]) / st["y_rng"]).mean(axis=1)


def recommend(maximize=None, mins=None, n_global=30000, n_local=6000, seed=0):
    st = _state()
    if maximize is not None and maximize not in TARGETS:
        maximize = None
    mins = {k: float(v) for k, v in (mins or {}).items()
            if v is not None and k in TARGETS}
    rng = np.random.default_rng(seed)

    def evalp(Xf):
        full = _to_full(Xf)
        preds = st["oracle"].predict(full)
        mask = np.ones(len(Xf), bool)
        for t, lo in mins.items():
            mask &= preds[:, TARGETS.index(t)] >= lo
        return full, preds, mask

    Xf = _sample_feasible(st, n_global, rng)
    full, preds, mask = evalp(Xf)
    feasible = bool(mask.any())
    score = _score(st, preds, maximize)
    if feasible:
        pool = np.where(mask)[0]
        best = pool[int(np.argmax(score[pool]))]
    else:
        viol = np.zeros(len(Xf))
        for t, lo in mins.items():
            j = TARGETS.index(t)
            viol += np.maximum(0.0, (lo - preds[:, j]) / st["y_rng"][j])
        best = int(np.argmin(viol))

    if feasible and n_local > 0:
        c = Xf[best]; w = 0.12 * (st["hi"] - st["lo"])
        Xf2 = _sample_feasible(st, n_local, rng,
                               np.maximum(st["lo"], c - w), np.minimum(st["hi"], c + w))
        full2, preds2, mask2 = evalp(Xf2)
        if mask2.any():
            s2 = _score(st, preds2, maximize)
            pool2 = np.where(mask2)[0]
            cand = pool2[int(np.argmax(s2[pool2]))]
            if s2[cand] > score[best]:
                full, preds, best = full2, preds2, cand

    predicted = {t: float(v) for t, v in zip(TARGETS, preds[best])}
    return {
        "maximize": maximize,
        "constraints": mins,
        "feasible": feasible,
        "satisfied": {t: bool(predicted[t] >= lo) for t, lo in mins.items()},
        "formulation": {n: float(v) for n, v in zip(FEATURES, full[best])},
        "predicted": predicted,
        "units": UNITS,
        "oracle_r2": st["r2"],
    }


def _find_asset(name):
    here = os.path.dirname(__file__)
    for p in (os.path.join(here, "..", name), os.path.join(os.getcwd(), name),
              name, os.path.join(here, name)):
        if os.path.exists(p):
            return p
    return None


# Vercel 단일 엔트리포인트가 모든 경로를 이 함수로 보내므로,
# 정적 파일(index.html 등)도 함수가 직접 서빙한다.
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/chart.umd.min.js": ("chart.umd.min.js", "application/javascript; charset=utf-8"),
    "/convergence.json": ("convergence.json", "application/json; charset=utf-8"),
}


class handler(BaseHTTPRequestHandler):
    def _raw(self, code, data, ctype):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, body):
        self._raw(code, json.dumps(body), "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in _ASSETS:
            fname, ctype = _ASSETS[path]
            fp = _find_asset(fname)
            if fp:
                with open(fp, "rb") as f:
                    return self._raw(200, f.read(), ctype)
            return self._json(404, {"error": f"{fname} not found"})
        if path == "/api/recommend":            # 헬스체크
            try:
                return self._json(200, {"ok": True, "oracle_r2": _state()["r2"]})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/api/recommend":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            out = recommend(maximize=req.get("maximize") or None,
                            mins=req.get("mins", {}) or {})
            self._json(200, out)
        except Exception as e:
            self._json(500, {"error": str(e)})
