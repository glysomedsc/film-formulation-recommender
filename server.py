# -*- coding: utf-8 -*-
"""
필름 배합 추천 웹 서버 (표준 라이브러리 http.server, 추가 설치 불필요).

  GET  /                    -> index.html (입력 화면)
  GET  /chart.umd.min.js    -> Chart.js (로컬 서빙)
  POST /recommend           -> {maximize, mins} 받아 최적 배합 계산 후 JSON
  GET  /convergence         -> BO vs 랜덤 수렴 곡선 데이터 (캐시)

실행:  .venv/Scripts/python.exe server.py   그리고 http://localhost:8000 접속
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from recommender import Recommender, TARGETS
import film_bo  # BO/랜덤 탐색 로직 재사용 (오라클은 아래에서 주입)

HERE = Path(__file__).parent
print("오라클 학습 중…")
REC = Recommender(str(HERE / "film_experiments.csv"))
print("준비 완료. 오라클 R2:", {k: round(v, 3) for k, v in REC.r2.items()})

_conv_cache = None
_conv_lock = threading.Lock()


def compute_convergence(budget=40, seeds=8):
    """이미 학습된 오라클로 BO vs 랜덤 수렴 곡선(여러 시드 평균) 계산."""
    global _conv_cache
    with _conv_lock:
        if _conv_cache is not None:
            return _conv_cache
        df = REC.df
        obj = film_bo.Objective(df)
        space = film_bo.SearchSpace(df)
        experiment = film_bo.make_experiment(REC.oracle, space, obj)
        base_best = float(max(obj.scalar(r) for r in df[TARGETS].values))

        bo, rd = [], []
        for s in range(seeds):
            bc, _, _ = film_bo.bayes_opt(experiment, space, budget, seed=s)
            rc, _, _ = film_bo.random_search(experiment, space, budget, seed=1000 + s)
            bo.append(bc); rd.append(rc)
        bo, rd = np.array(bo), np.array(rd)
        rand_final = float(rd.mean(0)[-1])
        bo_mean = bo.mean(0)
        reach = int(np.argmax(bo_mean >= rand_final)) + 1 if bo_mean[-1] >= rand_final else None

        _conv_cache = {
            "xs": list(range(1, budget + 1)),
            "bo_mean": bo_mean.round(4).tolist(),
            "bo_std": bo.std(0).round(4).tolist(),
            "rand_mean": rd.mean(0).round(4).tolist(),
            "rand_std": rd.std(0).round(4).tolist(),
            "base_best": round(base_best, 4),
            "rand_final": round(rand_final, 4),
            "bo_final": round(float(bo_mean[-1]), 4),
            "reach": reach,
            "seeds": seeds,
        }
        return _conv_cache


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")
        elif self.path == "/chart.umd.min.js":
            self._send(200, (HERE / "chart.umd.min.js").read_bytes(),
                       "application/javascript; charset=utf-8")
        elif self.path == "/convergence":
            self._send(200, json.dumps(compute_convergence()))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/recommend":
            self._send(404, json.dumps({"error": "not found"})); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            maximize = req.get("maximize") or None          # "" 또는 없음 -> None(균형)
            mins = req.get("mins", {}) or {}
            out = REC.recommend(maximize=maximize, mins=mins)
            self._send(200, json.dumps(out))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = 8000
    # 수렴 곡선은 서버 시작과 동시에 백그라운드로 미리 계산 (사용자 클릭 시 즉시 응답)
    threading.Thread(target=compute_convergence, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n▶ 웹 추천 시스템 실행 중:  http://localhost:{port}")
    print("  (수렴 곡선 백그라운드 계산 중… 잠시 후 준비됨 · 종료: Ctrl+C)")
    srv.serve_forever()
