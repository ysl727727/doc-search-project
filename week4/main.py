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

def build_tfidf(df, column_name):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(df[column_name])

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
        "query": "How can I create a list in one line using Python?",
        "relevant_doc_ids": ["D001"]
    },
    {
        "query": "How can I add extra functionality to a Python function?",
        "relevant_doc_ids": ["D010"]
    },
    {
        "query": "How do I fix merge conflicts in Git?",
        "relevant_doc_ids": ["D018"]
    },
    {
        "query": "How can I temporarily save my changes in Git?",
        "relevant_doc_ids": ["D019"]
    },
    {
        "query": "How does gradient descent optimize a machine learning model?",
        "relevant_doc_ids": ["D023"]
    },
    {
        "query": "What is the purpose of a loss function in machine learning?",
        "relevant_doc_ids": ["D024"]
    },
    {
        "query": "How are gradients calculated during neural network training?",
        "relevant_doc_ids": ["D030"]
    },
    {
        "query": "What is a neural network and how does it work?",
        "relevant_doc_ids": ["D031"]
    },
    {
        "query": "How do I multiply matrices using NumPy?",
        "relevant_doc_ids": ["D035"]
    },
    {
        "query": "How can I combine two DataFrames in pandas?",
        "relevant_doc_ids": ["D045"]
    },
    {
        "query": "Why should I use type hints in Python?",
        "relevant_doc_ids": ["D052"]
    },
    {
        "query": "What are convolutional neural networks used for?",
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
        relevant = item["relevant_doc_ids"]
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
        "Precision@3": precision_mean,
        "MRR": mrr_mean
    }

def analyze_failures(eval_set, search_func, k):
    fail_count = 0
    failures = []
    for item in eval_set:
        query = item["query"]
        relevant = item["relevant_doc_ids"]

        result = search_func(query)
        result_ids = result["doc_id"].tolist()
        
        rr = reciprocal_rank(
            result_ids,
            relevant
        )

        if rr == 0.0:
            fail_count += 1
            
            failures.append(
                (query, relevant, result_ids)
            )      

    if fail_count == 0:
        print("실패 케이스가 없습니다.")
    
    else:
        print(f"실패 케이스 : {fail_count}개\n")

        for query, relevant, result_ids in failures:
            
            print(f"질문: {query}")
            print(f"정답 doc_id : {relevant}")
            print(f"검색 결과 : {result_ids}")
            print()
                  
def main():
    df = load_data(DATA_PATH)
    df["content_clean"] = df["content"].apply(preprocess)
    df["title_clean"] = df["title"].apply(preprocess)

    df["content_weighted"] = (
        df["title_clean"] + " "
    ) * 3 + df["content_clean"]    
    
    # print("전처리 완료: content_clean 컬럼 생성") # 전처리 확인용

    print("\n=== 기본 TF-IDF ===")
    tfidf_matrix, vectorizer = build_tfidf(df,"content_clean")

    print("\n=== Weighted TF-IDF ===")
    weighted_matrix, weighted_vectorizer = build_tfidf(df,"content_weighted")

    print("\n=== 예시 검색: git merge conflicts ===")
    example = tfidf_search(
        "git merge conflicts",
        df,
        weighted_matrix,
        weighted_vectorizer
    )
    print(example)

    # print(f"\n평가셋 크기: {len(eval_set)}개 질문") # 기능1 확인용
    
    keyword = lambda q: keyword_search(q,df)

    tfidf = lambda q: tfidf_search(
        q,
        df,
        tfidf_matrix,
        vectorizer
    )

    weighted_tfidf = lambda q: tfidf_search(
        q,
        df,
        weighted_matrix,
        weighted_vectorizer
    )

    keyword_result = run_evaluation(
        eval_set,
        keyword,
        3
    )

    tfidf_result = run_evaluation(
        eval_set,
        tfidf,
        3
    )

    weighted_result = run_evaluation(
        eval_set,
        weighted_tfidf,
        3
    )

    print("\n=== 성능 비교 ===")
    print("                  Precision@3     MRR")

    print(
        f"Keyword Baseline      "
        f"{keyword_result['Precision@3']:.4f}     "
        f"{keyword_result['MRR']:.4f}"
    )

    print(
        f"TF-IDF                "
        f"{tfidf_result['Precision@3']:.4f}     "
        f"{tfidf_result['MRR']:.4f}"
    )

    print(
        f"Weighted TF-IDF       "
        f"{weighted_result['Precision@3']:.4f}     "
        f"{weighted_result['MRR']:.4f}"
    )

    print("\n=== 실패 케이스 (Weighted TF-IDF) ===") # Top-3 안에 정답 없음

    analyze_failures(
        eval_set,
        weighted_tfidf,
        3
    )

    # Baseline은 단어 개수만 비교하므로 관련 없는 문서가 포함될 수 있다.
    # TF-IDF는 중요한 단어에 가중치를 부여하여 더 관련성 높은 문서를 찾는다.

if __name__ == "__main__":
    main()