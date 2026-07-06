import pandas as pd
import numpy as np
import os
import sys
import re
DATA_PATH = "data/tech_docs.csv"

def load_data(file_path):
    if not os.path.exists(file_path):
        print("파일을 찾을 수 없습니다.")
        sys.exit()

    df = pd.read_csv(file_path)
    print(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
    return df

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cosine_similarity_numpy(a,b):
    dot_product = np.dot(a,b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)








def main():
    df = load_data(DATA_PATH)

    df = df.dropna(subset=["content"])
    df["content_clean"] = df["content"].apply(preprocess)
    print("전처리 완료 content_clean 칼럼 생성")
    # 전처리 확인방법 : print(df[["content", "content_clean"]].head(3)) 

 

if __name__ == "__main__":
    main()