import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re

da = pd.read_excel('다나와 상품.xlsx')
da = da.drop_duplicates(subset=['name.1'])

co = pd.read_excel('커뮤니티 상품 DB.xlsx')
co = co.drop_duplicates(subset=['title'])

def preprocess_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\s\d\w]', '', text)
    return ''

da['name.1'] = da['name.1'].apply(preprocess_text)
co['title'] = co['title'].apply(preprocess_text)

da['name.1'] = da['name.1'].fillna('').astype(str)
co['title'] = co['title'].fillna('').astype(str)

texts = da['name.1'].tolist() + co['title'].tolist()

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

from sklearn.cluster import DBSCAN

optimal_eps = 0.5  
min_samples = 1

dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X)

num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f'클러스터 수: {num_clusters}')

# 실루엣 스코어
if num_clusters > 1:
     score = silhouette_score(X, clusters)
     print(f'실루엣 스코어: {score}')
else:
     print('측정되지 않음')

clustered_texts = pd.DataFrame({'text': texts, 'cluster': clusters})
for cluster_num in range(num_clusters):
    print(f"\nCluster {cluster_num}:")
    cluster_texts = clustered_texts[clustered_texts['cluster'] == cluster_num]['text'].values
    for text in cluster_texts:  
        print(text)
    
with open('클러스터 결과 min_samples = 1.txt', 'w', encoding='utf-8') as file:
    for cluster_num in range(num_clusters):
        file.write(f"\nCluster {cluster_num}:\n")
        cluster_texts = clustered_texts[clustered_texts['cluster'] == cluster_num]['text'].values
        for text in cluster_texts:
            file.write(text + '\n')

