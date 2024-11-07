import pandas as pd
import numpy as np
import pytrie
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드
da = pd.read_excel('다나와 상품.xlsx')
da = da.drop_duplicates(subset=['name.1'])

co = pd.read_excel('커뮤니티 상품 DB.xlsx')
co = co.drop_duplicates(subset=['title'])

# 텍스트 전처리 함수
def preprocess_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\s\d\w]', '', text)
    return ''

# 전처리 적용
da['name.1'] = da['name.1'].apply(preprocess_text)
co['title'] = co['title'].apply(preprocess_text)

# Null 값 처리
da['name.1'] = da['name.1'].fillna('').astype(str)
co['title'] = co['title'].fillna('').astype(str)

# Trie 구축 함수
def build_trie_with_prefixes(strings):
    trie = pytrie.StringTrie()
    for string in strings:
        if isinstance(string, str):
            for i in range(1, len(string) + 1):
                prefix = string[:i]
                trie[prefix] = string
    return trie

# 다나와 상품 목록으로 Trie 구조 생성
da_list = da['name.1'].tolist()
trie = build_trie_with_prefixes(da_list)

# TF-IDF 벡터라이저 설정 및 변환
vectorizer = TfidfVectorizer()
da_tfidf = vectorizer.fit_transform(da['name.1'])
co_tfidf = vectorizer.transform(co['title'])

def similar_posts(query_vector, trie, threshold=0.8):
    similar_posts = []
    for prefix in trie:
        trie_value = trie[prefix]
        trie_vector = vectorizer.transform([trie_value])
        similarity = cosine_similarity(query_vector, trie_vector).flatten()[0]
        if similarity >= threshold:
            similar_posts.append((trie_value, similarity))
    return similar_posts

# 커뮤니티 게시글 매칭 함수
def match_post():
    results = []

    def task(row, query_vector):
        similar_matches = similar_posts(query_vector, trie)
        return [(row['title'], match[0], match[1]) for match in similar_matches]

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(task, row, co_tfidf[i]): row for i, row in co.iterrows()}

        for future in futures:
            try:
                similar_matches = future.result()
                for match in similar_matches:
                    results.append({
                        '커뮤니티 상품': match[0],
                        '다나와 상품': match[1],
                        '코사인 유사도': match[2]
                    })
            except Exception as e:
                print(f"Error processing {futures[future]['title']}: {e}")

    return pd.DataFrame(results)

matching = match_post()
matching.to_excel('코사인 매칭결과.xlsx', index=False)
print("Matching results saved to '코사인 매칭결과.xlsx'")
