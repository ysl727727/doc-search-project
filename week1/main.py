import pandas as pd
import numpy as np
import os
import sys

def load_data(file_path):
    if not os.path.exists(file_path):
        print("파일을 찾을 수 없습니다.")
        sys.exit()

    df = pd.read_csv(file_path)
    print(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
    return df

df = load_data("data/tech_docs.csv")

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

explore_structure(df)

def show_category_distribution(df):
    print("=====카테고리 분포=====")

    category_counts = df["category"].value_counts()
    total_docs = len(df)

    for category, count in category_counts.items():
        ratio = (count / total_docs) * 100
        print(f"{category} : {count}개 ({ratio:.1f}%)")
        docs = df[df["category"] == category]
        word_counts = []

        for text in docs["content"]:
            word_counts.append(len(text.split()))

        avg_words = np.mean(word_counts)

        print(f"평균 단어 수 :{avg_words:.1f}")
        print()
    
    category_dict = category_counts.to_dict()
    return category_dict

show_category_distribution(df)

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

check_missing(df)