import pandas as pd
import numpy as np
import pytrie
import re
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
# font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
# font_name = fm.FontProperties(fname=font_path).get_name()
# plt.rc('font', family=font_name)

# 데이터 불러오기
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

def build_trie_with_prefixes(strings):
    trie = pytrie.StringTrie()
    for string in strings:
        if isinstance(string, str):
            for i in range(1, len(string) + 1):
                prefix = string[:i]
                trie[prefix] = string
    return trie

# 다나와 to 트라이 구조
da_list = da['name.1'].tolist()
trie = build_trie_with_prefixes(da_list)

# 자카드 유사도 계산 함수
def jaccard(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union

def similar_posts(query, trie, threshold=0.8):
    query_set = set(query.split())
    similar_posts = []
    for prefix in query_set:
        if prefix in trie:
            trie_value = trie[prefix]
            trie_set = set(trie_value.split())
            similarity = jaccard(query_set, trie_set)
            if similarity >= threshold:
                similar_posts.append((trie_value, similarity))
    return similar_posts

# 커뮤니티 게시글 빈칸 기준으로 나누어 전처리
co['processed'] = co['title'].apply(lambda x: x.split() if isinstance(x, str) else [])

def match_post():
    results = []

    def task(row):
        community_title = ' '.join(row['processed'])  # 리스트를 다시 문자열로 변환
        similar_matches = similar_posts(community_title, trie)
        return [(community_title, match[0], match[1]) for match in similar_matches]

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(task, row): row for _, row in co.iterrows()}

        for future in futures:
            try:
                similar_matches = future.result()
                for match in similar_matches:
                    results.append({
                        '커뮤니티 상품': match[0],
                        '다나와 상품': match[1],
                        '자카드 유사도': match[2]
                    })
            except Exception as e:
                print(f"Error processing {futures[future]['title']}: {e}")

    return pd.DataFrame(results)

matching = match_post()
print(matching)

# TextRank 그래프 구축 및 시각화
def textrRank(matching_df):
    G = nx.Graph()

    # 노드 추가 및 가중치 설정
    for index, row in matching_df.iterrows():
        community_product = row['커뮤니티 상품']
        danawa_product = row['다나와 상품']
        similarity = row['자카드 유사도']
        
        if not G.has_node(community_product):
            G.add_node(community_product, label=community_product)
        if not G.has_node(danawa_product):
            G.add_node(danawa_product, label=danawa_product)
        
        G.add_edge(community_product, danawa_product, weight=similarity)

    # 페이지랭크 계산
    pagerank = nx.pagerank(G, weight='weight')

    # 노드 크기를 페이지랭크에 따라 조정
    node_size = [pagerank[node] * 10000 for node in G]

    # 그래프 그리기
    pos = nx.spring_layout(G)
    plt.figure(figsize=(15, 10))
    nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='skyblue', edge_color='gray', font_size=10, font_weight='bold')
    plt.title("TextRank 시각화")
    plt.show()

textrRank(matching)

# 클러스터링
def communities(matching_df):
    G = nx.Graph()

    # 노드 & 엣지
    for _, row in matching_df.iterrows():
        G.add_node(row['커뮤니티 상품'], type='커뮤니티')
        G.add_node(row['다나와 상품'], type='다나와')
        G.add_edge(row['커뮤니티 상품'], row['다나와 상품'], weight=row['자카드 유사도'])

    # 커뮤니티 탐지 (클러스터링)
    communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))

    # 노드에 클러스터 정보 추가
    node_colors = {}
    for i, community in enumerate(communities):
        for node in community:
            node_colors[node] = i

    
    colors = [plt.cm.tab10(i / float(len(communities))) for i in range(len(communities))]

    
    pos = nx.spring_layout(G, seed=42)  

    plt.figure(figsize=(15, 10))

    # 노드
    for node, (x, y) in pos.items():
        plt.scatter(x, y, color=colors[node_colors[node]], s=100, alpha=0.6, label=node_colors[node])

    # 엣지
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], alpha=0.5, width=edge[2]['weight'] * 5, edge_color='gray')

    # 노드 라벨
    nx.draw_networkx_labels(G, pos, font_size=10, font_family=plt.rcParams['font.family'])

    # 범례
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', title='클러스터')

    plt.title('클러스터링 시각화', fontsize=15)
    plt.axis('off')
    plt.show()

communities(matching)
