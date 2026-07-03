import pandas as pd
import numpy as np
import os
import sys
DATA_PATH = "data/tech_docs.csv"

def load_data(file_path):
    if not os.path.exists(file_path):
        print("파일을 찾을 수 없습니다.")
        sys.exit()

    df = pd.read_csv(file_path)
    print(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
    return df


def explore_structure(df):
    rows, cols = df.shape

    print("=====데이터 구조=====")
    print(f"행: {rows}")
    print(f"열: {cols}")

    print("=====칼럼명=====")
    for col in df.columns:
        print(col)

    print("=====자료형=====")
    print(df.dtypes)
    
    print("=====상위 5행=====")
    print(df.head())


def show_category_distribution(df):
    print("=====카테고리 분포=====")

    category_counts = df["category"].value_counts()
    total_docs = len(df)
    category_info = {}

    for category, count in category_counts.items():
        ratio = (count / total_docs) * 100
        print(f"{category} : {count}개 ({ratio:.1f}%)")
        
        docs = df[df["category"] == category]
        word_counts = []

        for text in docs["content"]:
            word_counts.append(len(text.split()))

        avg_words = np.mean(word_counts)

        print(f"{category} 평균 단어 수 :{avg_words:.1f}\n")
    
        category_info[category] = {
            "count": count,
            "ratio": ratio,
            "avg_words": avg_words
        }

    return category_info


def check_missing(df):
    print("=====결측치 현황=====")

    missing = df.isnull().sum()
    missing_ratio = (missing / len(df)) * 100
    
    no_missing = []
    missing_columns = []

    for col in df.columns:

        if missing[col] == 0:
            no_missing.append(col)

        if missing[col] > 0:

            missing_columns.append(col)   
        
            ratio = missing_ratio[col]
            
            if ratio < 5:
                severity = "낮음"
            elif ratio < 20:
                severity = "주의"            
            else:
                severity = "높음"
            
            print(f"{col}: {missing[col]}개 ({ratio:.1f}%) - {severity}")          
    
    if len(missing_columns) == 0:
        print("결측치가 있는 컬럼: 없음")
    else:
        print("결측치가 있는 컬럼:", missing_columns)

    print()
    print("결측치 없는 컬럼:")
    print(no_missing)

    return missing.to_dict()

def numpy_doc_stats(df):
    contents = df["content"].dropna()
    lengths = [len(text.split()) for text in contents]
    doc_lengths = np.array(lengths)

    mean_lengths = np.mean(doc_lengths)
    std_lengths = np.std(doc_lengths, ddof=1)
    median_lengths = np.median(doc_lengths)
    min_lengths = np.min(doc_lengths)
    max_lengths = np.max(doc_lengths)

    print("\n===== Numpy 문서 길이 통계 =====")
    print(f"평균 단어 수 : {mean_lengths:.1f}")
    print(f"표준편차 : {std_lengths:.1f}")
    print(f"중앙값 : {median_lengths:.1f}")
    print(f"최솟값 : {min_lengths}")
    print(f"최댓값 : {max_lengths}")
    
    short_docs = doc_lengths[doc_lengths < 50]

    print(f"\n50단어 미만 문서 수 {len(short_docs)}")
    print(short_docs)

    pandas_lengths = df["content"].dropna().apply(lambda text: len(text.split()))

    pandas_mean = pandas_lengths.mean()
    pandas_std = pandas_lengths.std()
    pandas_median = pandas_lengths.median()
    pandas_min = pandas_lengths.min()
    pandas_max = pandas_lengths.max()

    print("\n===== pandas와 NumPy 비교 =====")

    print(f"평균 : NumPy={mean_lengths:.1f}, pandas={pandas_mean:.1f}")
    print(f"일치 여부 : {np.isclose(mean_lengths, pandas_mean)}\n")

    print(f"표준편차 : NumPy={std_lengths:.1f}, pandas={pandas_std:.1f}")
    print(f"일치 여부 : {np.isclose(std_lengths, pandas_std)}\n")

    print(f"중앙값 : NumPy={median_lengths:.1f}, pandas={pandas_median:.1f}")
    print(f"일치 여부 : {np.isclose(median_lengths, pandas_median)}\n")

    print(f"최솟값 : NumPy={min_lengths}, pandas={pandas_min}")
    print(f"일치 여부 : {np.isclose(min_lengths, pandas_min)}\n")

    print(f"최댓값 : NumPy={max_lengths}, pandas={pandas_max}")
    print(f"일치 여부 : {np.isclose(max_lengths, pandas_max)}\n")

def main():
    df = load_data(DATA_PATH)

    explore_structure(df)
    show_category_distribution(df)
    check_missing(df)
    numpy_doc_stats(df)

if __name__ == "__main__":
    main()