# Doc Search Project

프로젝트 마일스톤 과제입니다.

## Week1 기능

* 데이터셋 로드
* 데이터 구조 확인
* 카테고리별 문서 개수 및 평균 단어 수 분석
* 결측치 검사
* NumPy와 pandas를 이용한 문서 길이 통계 비교

## Week2 기능

* 텍스트 전처리 (소문자 변환, 특수문자 제거, 공백 정리)
* NumPy를 이용한 코사인 유사도 직접 구현
* Keyword Baseline 검색
* TF-IDF 벡터화
* TF-IDF 기반 Top-3 문서 검색
* Baseline과 TF-IDF 검색 결과 비교

## Week3 기능

* 검색 평가용 Evaluation Set 구성
* Precision@3 계산
* MRR(Mean Reciprocal Rank) 계산
* Keyword Baseline과 TF-IDF 검색 성능 비교
* 검색 실패 케이스 분석

## Week4 기능

* 전체 검색 파이프라인 통합
* CSV 데이터 로드 → 전처리 → TF-IDF 벡터화 → 검색 → 평가 → 실패 케이스 분석
* TF-IDF 예시 검색 실행
* Baseline, TF-IDF 성능 비교
* (선택) 제목 가중치(Title Weighting)를 적용한 TF-IDF 성능 비교

## 프로젝트 구조

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

## 실행 방법

### Week1

```bash
python week1/main.py
```

### Week2

```bash
python week2/main.py
```

### Week3

```bash
python week3/main.py
```

### Week4

```bash
python week4/main.py
```

## 사용 라이브러리

* pandas
* NumPy
* scikit-learn
* re (Python 기본 라이브러리)

## 학습 내용

* pandas를 이용한 데이터 처리
* NumPy를 이용한 벡터 연산
* 텍스트 전처리
* TF-IDF 기반 문서 벡터화
* 코사인 유사도 계산
* 정보 검색(Search) 기초
* Precision@k, MRR을 이용한 검색 성능 평가
* 검색 파이프라인 통합
* 제목 가중치를 활용한 검색 성능 개선 실험