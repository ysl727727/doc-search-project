# 🔍 Doc Search Project

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

> 기술 문서 60건을 대상으로 Keyword Baseline과 TF-IDF 검색을 직접 구현하고, Precision@3와 MRR로 성능을 비교한 4주 마일스톤 과제입니다.

## 📌 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| 기간 | 2026-06-29 \~ 2026-07-19 (Week1 \~ Week4) |
| 데이터 | `data/tech_docs.csv` — 기술 문서 60건 × 5열 (Python · Git · AI기초 · NumPy · pandas) |
| 검색 방법 | Keyword Baseline → TF-IDF → 제목 가중치 TF-IDF |
| 평가 | 평가 질문 12개, Precision@3 · MRR |

### 최종 결과 (`week4/main.py`)

| 방법 | Precision@3 | MRR |
| --- | ---: | ---: |
| Keyword Baseline | 0.2778 | 0.6528 |
| TF-IDF | **0.3333** | 0.7500 |
| Weighted TF-IDF (제목 가중치) | 0.3056 | **0.8194** |

Precision@3는 기본 TF-IDF가, MRR은 제목 가중치 TF-IDF가 가장 높았습니다. 지표 하나만 보지 않고 검색 목적에 맞는 지표로 방법을 골라야 한다는 점을 확인했습니다.

## 🗓️ 주차별 구현 내용

### Week1 — 데이터 탐색

- 데이터셋 로드
- 데이터 구조 확인
- 카테고리별 문서 개수 및 평균 단어 수 분석
- 결측치 검사
- NumPy와 pandas를 이용한 문서 길이 통계 비교

### Week2 — 전처리와 검색

- 텍스트 전처리 (소문자 변환, 특수문자 제거, 공백 정리)
- NumPy를 이용한 코사인 유사도 직접 구현
- Keyword Baseline 검색
- TF-IDF 벡터화
- TF-IDF 기반 Top-3 문서 검색
- Baseline과 TF-IDF 검색 결과 비교

### Week3 — 검색 평가

- 검색 평가용 Evaluation Set 구성
- Precision@3 계산
- MRR(Mean Reciprocal Rank) 계산
- Keyword Baseline과 TF-IDF 검색 성능 비교
- 검색 실패 케이스 분석

### Week4 — 파이프라인 통합

- 전체 검색 파이프라인 통합
- CSV 데이터 로드 → 전처리 → TF-IDF 벡터화 → 검색 → 평가 → 실패 케이스 분석
- TF-IDF 예시 검색 실행
- Baseline, TF-IDF 성능 비교
- (선택) 제목 가중치(Title Weighting)를 적용한 TF-IDF 성능 비교

## 🗂️ 프로젝트 구조

```text
doc-search-project/
│
├── data/
│   └── tech_docs.csv
├── week1/
│   └── main.py
├── week2/
│   └── main.py
├── week3/
│   └── main.py
├── week4/
│   └── main.py
└── README.md
```

## ▶️ 실행 방법

```bash
python week1/main.py   # Week1
python week2/main.py   # Week2
python week3/main.py   # Week3
python week4/main.py   # Week4
```

## 🧰 사용 라이브러리

- pandas
- NumPy
- scikit-learn
- re (Python 기본 라이브러리)

## 📚 학습 내용

- pandas를 이용한 데이터 처리
- NumPy를 이용한 벡터 연산
- 텍스트 전처리
- TF-IDF 기반 문서 벡터화
- 코사인 유사도 계산
- 정보 검색(Search) 기초
- Precision@k, MRR을 이용한 검색 성능 평가
- 검색 파이프라인 통합
- 제목 가중치를 활용한 검색 성능 개선 실험

---

<sub>[Yongseok Lee](https://github.com/ysl727727) · 학습 기록은 [TIL](https://github.com/ysl727727/TIL)에 정리하고 있습니다.</sub>
