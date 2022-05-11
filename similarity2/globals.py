"""
全局变量有关代码
"""
from similarity.word2vec_similarity_catalog import model as similarity1_model
import torch
import os
import pathlib
import configparser

# gensim 词向量模型
MODEL = similarity1_model
# 是否启用GPU设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("正在使用独立GPU")
else:
    print("正在使用CPU")
# 路径
ROOT_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)  # 路径为 E:\PythonProject\Similarity
CONFIG_PATH = os.path.join(ROOT_PATH, 'config.ini')
DATA_PATH = os.path.join(ROOT_PATH, 'similarity/data/')
RESULT_PATH = os.path.join(ROOT_PATH, 'similarity/result/')
MODEL_PATH = os.path.join(ROOT_PATH, 'similarity/model/')

# config.ini 设置
CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_PATH, encoding='utf8')
