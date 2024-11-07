import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# 주어진 상품명 리스트
product_names = [
"원더랜드 그랜드 마스터2",
"스틸시리즈 아크티스 로우"
]


# Convert product_names array to DataFrame
data_cleaned = pd.DataFrame(product_names, columns=['title'])

# Strip any leading/trailing whitespace characters from titles
data_cleaned['title'] = data_cleaned['title'].str.strip()

# Vectorize the titles using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data_cleaned['title'])

# Perform MiniBatchKMeans clustering with smaller batch size
wcss = []
for i in range(10000, 30001, 5000):  # Checking for 1000 to 10000 clusters in steps of 1000
    mbk = MiniBatchKMeans(n_clusters=i, random_state=42, batch_size=100)
    mbk.fit(X)
    wcss.append(mbk.inertia_)

# Plot the WCSS to find the Elbow point
plt.figure(figsize=(10, 5))
plt.plot(range(10000, 30000, 2500), wcss, marker='o')
plt.title('Elbow Method For Optimal K with MiniBatchKMeans')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()