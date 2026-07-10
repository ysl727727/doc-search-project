# Doc Search Project

프로젝트 마일스톤 과제입니다.

## Week1 기능

- 데이터셋 로드
- 데이터 구조 확인
- 카테고리별 문서 개수 및 평균 단어 수 분석
- 결측치 검사
- NumPy와 pandas를 이용한 문서 길이 통계 비교

## Week2 기능

- 텍스트 전처리 (소문자 변환, 특수문자 제거, 공백 정리)
- NumPy를 이용한 코사인 유사도 직접 구현
- Keyword Baseline 검색
- TF-IDF 벡터화
- TF-IDF 기반 Top-3 문서 검색
- Baseline과 TF-IDF 검색 결과 비교

## 프로젝트 구조

```
doc-search-project/
│
├── data/
│   └── tech_docs.csv
├── week1/
│   └── main.py
├── week2/
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

## 사용 라이브러리

- pandas
- NumPy
- scikit-learn
- re (Python 기본 라이브러리)