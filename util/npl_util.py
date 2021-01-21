# -*- coding: utf-8 -*-
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import logging
import re

util_logger = logging.getLogger('RATE.npl')


class WordVector(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            util_logger.info('loading google pretrained word2vec model')
            cls._instance = \
                KeyedVectors.load_word2vec_format(
                    '../NAECF_cos/data/GoogleNews-vectors-negative300.bin',
                    limit=int(1e4),
                    binary=True)
            util_logger.info('loaded')
        return cls._instance


def word2vec(word: str):
    """
    返回单词对应词向量，若无则返回单词
    :param word:
    :return:
    """
    try:
        return WordVector()[word]
    except KeyError:
        return word


def word2id(word: str):
    """
    返回单词对应id，若无则返回-1
    :param word:
    :return:
    """
    try:
        return WordVector().vocab[word].index
    except KeyError:
        return -1


def words2vectors(words: list):
    """
    把单词列表转换为对应的词向量列表，并删除没有对应词向量的单词
    :param words:
    :return: numpy.array
    """
    vec = [word2vec(x) for x in words]
    return np.array(list(filter(lambda x: type(x) != str, vec)))


def words2ids(words: list):
    """
    把单词列表转换为对应的词token id，并删除没有对应词向量的单词
    :param words:
    :return: list(int)
    """
    ids = [word2id(x) for x in words]
    return list(filter(lambda x: x >= 0, ids))


def get_unused_words(words: list):
    """
    返回在word2vec中没有对应向量的单词
    :param words:
    :return: list(str)
    """
    vec = [word2vec(x) for x in words]
    return list(filter(lambda x: type(x) == str, vec))


def words2vectors_fixed_length(words: list, words_length: int):
    """
    把单词列表转换为对应的词向量列表，并删除没有对应词向量的单词,
    根据words_length删掉多余单词向量，或者补充零向量
    :param words:
    :param words_length:
    :return: numpy.array shape:(words_length, vec_dim)
    """
    vectors = words2vectors(words)
    if len(vectors) > words_length:
        return vectors[:words_length]
    elif len(vectors) < words_length:
        diff = words_length - len(vectors)
        return np.pad(vectors,
                      ((0, diff), (0, 0)),
                      'constant')
    else:
        return vectors


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r' \( ', string)
    string = re.sub(r"\)", r" \) ", string)
    string = re.sub(r"\?", r" \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def split_sentence_to_word_list(sent: str):
    return clean_str(sent).split()
