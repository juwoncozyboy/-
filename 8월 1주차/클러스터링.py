import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re

# 텍스트 파일 경로
file_path = 'unique_replated_title-test.txt'

# 텍스트 파일을 읽어서 텍스트 자체로 가져오기
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()



def preprocess_text(text):
    # 입력이 문자열인지 확인
    if isinstance(text, str):
        # 특수문자 제거: 알파벳, 숫자, 공백만 남기기
        return re.sub(r'[^\s\d\w]', '', text)
    return ''


text = preprocess_text(text)
# print(text)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.positions = []
        self.frequency = 0

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.positions = []
        self.frequency = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word, position):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.positions.append(position)
        node.frequency += 1
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_end_of_word:
            return node.positions, node.frequency
        return None

    def _collect_all_words(self, node, prefix, result):
        if node.is_end_of_word:
            result[prefix] = node.frequency
        for char, child_node in node.children.items():
            self._collect_all_words(child_node, prefix + char, result)
    
    def get_all_word_frequencies(self):
        result = {}
        self._collect_all_words(self.root, '', result)
        return result
# Trie 생성 및 단어 삽입
trie = Trie()


words = text.split()
for i, word in enumerate(words):
    position = text.find(word)
    trie.insert(word, position)

# 모든 단어의 빈도 출력
word_frequencies = trie.get_all_word_frequencies()
for word, frequency in word_frequencies.items():
    print(f"Word '{word}' has frequency: {frequency}")
