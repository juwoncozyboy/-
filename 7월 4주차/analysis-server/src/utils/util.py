import re

from PyKomoran import *
from pykospacing import Spacing

import src.utils.constant as constant

spacing = Spacing()
komoran = Komoran(DEFAULT_MODEL["LIGHT"])


def cleaning(text: str):
    text = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "", text)
    text = re.sub(r"(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", " ", text)

    for p in constant.symbol_mapping:
        text = text.replace(p, constant.symbol_mapping[p])
    for p in constant.symbol:
        text = text.replace(p, "")

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub("([ㄱ-ㅎㅏ-ㅣ]+)", "", text)
    text = re.sub("[^\w\s\n]", "", text)
    return text


def spacingWord(text: str):
    new_sentence = text.replace(" ", "")  # 띄어쓰기가 없는 문장 임의로 만들기
    kospacing_sent = spacing(new_sentence)
    return kospacing_sent


def lemmatize(sentence):
    morphtags = komoran.pos(sentence)
    words = []
    for m, t in enumerate(morphtags):
        k = t.get_pos()
        if k == "NNP" or k == "NNG":
            words.append(t.get_morph())
        elif k == "VA" or k == "VV":
            words.append(t.get_morph() + "다")
    return words
