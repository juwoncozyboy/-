from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 형태소 분석기 초기화
okt = Okt()

# 예시 데이터
titles = ["신형 노트북 할인", "스마트폰 특가", "여름 신발 세일"]
products = ["노트북", "스마트폰", "신발", "TV", "냉장고"]

# 형태소 분석 함수
def tokenize(text):
    return ' '.join(okt.nouns(text))

# 형태소 분석 적용
tokenized_titles = [tokenize(title) for title in titles]
tokenized_products = [tokenize(product) for product in products]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
title_vectors = vectorizer.fit_transform(tokenized_titles)
product_vectors = vectorizer.transform(tokenized_products)

# 유사도 계산
similarity_matrix = cosine_similarity(title_vectors, product_vectors)

# 매칭 결과 출력
for i, title in enumerate(titles):
    best_match_idx = similarity_matrix[i].argmax()
    best_match_product = products[best_match_idx]
    similarity_score = similarity_matrix[i][best_match_idx]
    print(f"'{title}' => '{best_match_product}' (유사도: {similarity_score:.2f})")