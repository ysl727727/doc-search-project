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

    print(f"TF-IDF 행렬 크기: ({rows}, {cols}) | 사용된 단어 수: {cols}")

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
       
    top_indices = np.array(similarities).argsort()[::-1][:top_k]
    result = df.iloc[top_indices].copy()
    result["similarity"] = np.array(similarities)[top_indices].round(4)

    return result[["doc_id", "title", "category", "similarity"]]
    
eval_set = [
    {
        "query": "python list comprehension",
        "relevant_doc_ids": ["D001"]
    },
    {
        "query": "python decorators explained",
        "relevant_doc_ids": ["D010"]
    },
    {
        "query": "git merge conflict",
        "relevant_doc_ids": ["D018"]
    },
    {
        "query": "git stash changes",
        "relevant_doc_ids": ["D019"]
    },
    {
        "query": "what is gradient descent",
        "relevant_doc_ids": ["D023"]
    },
    {
        "query": "loss function machine learning",
        "relevant_doc_ids": ["D024"]
    },
    {
        "query": "backpropagation algorithm",
        "relevant_doc_ids": ["D030"]
    },
    {
        "query": "neural networks introduction",
        "relevant_doc_ids": ["D031"]
    },
    {
        "query": "numpy matrix operations",
        "relevant_doc_ids": ["D035"]
    },
    {
        "query": "pandas merge dataframe",
        "relevant_doc_ids": ["D045"]
    },
    {
        "query": "python type hints",
        "relevant_doc_ids": ["D052"]
    },
    {
        "query": "convolutional neural networks",
        "relevant_doc_ids": ["D060"]
    }
]

def precision_at_k(result_ids, relevant_doc_ids, k):
    top_k = result_ids[:k]
    correct = len(set(top_k) & set(relevant_doc_ids))
    return correct / k

def reciprocal_rank(result_ids, relevant_doc_ids):
    for rank, doc_id in enumerate(result_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1 / rank
    return 0.0

def run_evaluation(eval_set, search_func, k):
    precision_scores = []
    mrr_scores = []
    for item in eval_set:
        query =  item["query"]
        relevant = item["relevant_doc_idc"]
        result = search_func(query)
        result_ids = result["doc_id"].tolist()
        
        precision = precision_at_k(
            result_ids,
            relevant,
            k
        )

        mrr = reciprocal_rank(
            result_ids,
            relevant
        )

        precision_scores.append(precision)
        mrr_scores.append(mrr)

    precision_mean = np.mean(precision_scores)
    mrr_mean = np.mean(mrr_scores)

    return{
        "precision@3": precision_mean,
        "MRR": mrr_mean
    }


def main():
    df = load_data(DATA_PATH)
    df["content_clean"] = df["content"].apply(preprocess)
    print("전처리 완료: content_clean 컬럼 생성")
    # 전처리 확인방법 : print(df[["content", "content_clean"]].head(3)) 
   
    tfidf_matrix, vectorizer = build_tfidf(df) 

    print(f"\n평가셋 크기: {len(eval_set)}개 질문") # 기능1 확인
    
    result = precision_at_k(
        ["D001", "D012", "D059"],
        ["D001", "D059"],
        3
    )
    print(f"{result:.3f}")


    query = "how does gradient descent work in machine learning"
    
    print(f"\n질문: {query}")

    print("\n=== Keyword Baseline ===")
    print(keyword_search(query, df))
    
    print("\n=== TF-IDF Search ===")
    print(tfidf_search(query, df, tfidf_matrix, vectorizer))

    # Baseline은 단어 개수만 비교하므로 관련 없는 문서가 포함될 수 있다.
    # TF-IDF는 중요한 단어에 가중치를 부여하여 더 관련성 높은 문서를 찾는다.

if __name__ == "__main__":
    main()