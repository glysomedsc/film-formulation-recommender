# 필름 배합 추천 시스템 (Film Formulation Recommender)

소재 R&D는 **탐색 게임**입니다. 배합을 만들고 물성을 재는 실험 한 번에 시간·비용이 크고,
배합 공간은 연속·다차원이라 전수조사가 불가능합니다.
**적은 실험으로 최적 배합에 도달하는 것**이 곧 경쟁력입니다.

이 프로젝트는 조성·공정 조건에서 물성을 예측하는 회귀 모델을 만들고,
그 위에서 **베이지안 최적화(BO)**로 최적 배합을 최소 실험으로 탐색합니다.
사용자는 웹 화면에서 원하는 물성 조건만 입력하면, 시스템이 조건을 만족하는
최적 배합·공정 조건을 계산해 추천합니다.

## 문제 정의

- **입력 7** — `resin_A_ratio`, `resin_B_ratio`, `plasticizer`, `compatibilizer` (조성 4종, 합≈100%),
  `uv_stabilizer`(phr), `process_temp`(℃), `draw_ratio`
- **타깃 3** — `tensile_strength`(인장강도), `transparency`(투명도), `barrier`(차단성)
- 물성 3개는 서로 **트레이드오프**(예: 투명도↔차단성 r≈−0.72) → 무엇을 우선할지 먼저 정함

## 정량 목표 & 결과

| 목표 | 결과 |
|------|------|
| 회귀 모델 R² ≥ 0.9 | HistGBM 오라클 — 인장 **0.944** · 투명 **0.969** · 차단 **0.972** |
| 랜덤보다 적은 실험으로 더 높은 물성 | BO가 **~11회** 만에 랜덤 40회 성과 도달 (수렴 곡선 입증) |

## 핵심 설계

- **오라클(대리 실험):** 학습된 회귀모델이 "실험 1회 = 1회 쿼리"로 물성을 예측
- **표본 효율 탐색:** GP 대리모델 + **EI 획득함수**로 다음 실험점 선택,
  exploration↔exploitation 균형을 `xi`로 명시 제어 (`film_bo.py`)
- **제약 처리:** 조성 합=100은 `compatibilizer`를 잔여량으로 흡수, uv_stabilizer(phr)는 독립 변수
- **추천 엔진:** 사용자 제약(하한 ≥)을 만족하며 지정 물성을 최대화(또는 균형 최적)하도록
  조밀 전역탐색 + 국소 정밀화 (`recommender.py`)

## 파일 구성

| 파일 | 역할 |
|------|------|
| `film_bo.py` | BO vs 랜덤 표본효율 분석 (수렴 곡선 `bo_vs_random.png` 생성) |
| `recommender.py` | 오라클 학습 + 제약 최적화 추천 엔진 |
| `server.py` | 웹 서버 (표준 라이브러리 `http.server`, 추가 설치 불필요) |
| `index.html` | 웹 화면 — 히어로 / 조건 입력·추천 / 표본효율 차트 (Chart.js) |
| `chart.umd.min.js` | Chart.js (로컬 서빙, 오프라인 동작) |
| `film_experiments.csv` | 실험 데이터 (600행) |
| `필름조성예측.ipynb` | 초기 탐색 노트북 |

## 실행 방법

```bash
# 의존성: numpy, pandas, scikit-learn, scipy, matplotlib
python server.py
# 브라우저에서 http://localhost:8000 접속
```

- **① 배합 추천** — 물성 하한(≥)과 최대화 대상(선택) 입력 → 추천 배합·예상 물성 + 차트
- **② 표본 효율 분석** — BO vs 랜덤 수렴 곡선 (Chart.js)
- **③ 모델 정보** — 오라클 R², 엔진 설명

배치 분석만 보려면: `python film_bo.py` → `bo_vs_random.png` 생성.
