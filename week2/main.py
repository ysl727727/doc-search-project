import pandas as pd
import numpy as np
import os
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer
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

def keyword_search(query, df, top_k=3):
    query = preprocess(query)
    query_words = set(query.split())

    scores = []

    for content in df["content_clean"]:
        doc_words = set(content.split())
        score = len(query_words & doc_words)
        scores.append(score)

    df["score"] = scores

    result = df.sort_values("score", ascending=False)
    result = result.head(top_k)

    return result[["doc_id", "title", "category", "score"]]

def build_tfidf(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(df["content_clean"])

    rows, cols = tfidf_matrix.shape

    print(f"TF-IDF 행렬 크기: ({rows}, {cols}) | 사용된 단어 수:{cols}")

    return tfidf_matrix, vectorizer

def tfidf_search(query, df, tfidf_matrix, vectorizer, top_k=3):
    query = preprocess(query)
    query_vector = vectorizer.transform([query])
    query_vector = query_vector.toarray()[0]

    similarities = []

    for i in range(len(df)):
        doc_vector = tfidf_matrix[i].toarray()[0]
        similarity = cosine_similarity_numpy(query_vector, doc_vector)
        similarities.append(similarity)
        top_indices = np.array(similarities).argsort()[::1][:top_k]
        
        result = df.iloc[top_indices].copy()
        result["similarity"] = np.array(similarities)[top_indices]

        return result[["doc_id", "title", "category", "similarity"]]
    


    



def main():
    df = load_data(DATA_PATH)

    df = df.dropna(subset=["content"])
    df["content_clean"] = df["content"].apply(preprocess)
    print("전처리 완료 content_clean 칼럼 생성")
    # 전처리 확인방법 : print(df[["content", "content_clean"]].head(3)) 

    # 기능3 keyword_search main에 넣어야함 

 

if __name__ == "__main__":
    main()