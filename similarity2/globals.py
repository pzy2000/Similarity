"""
全局变量有关代码
"""
import torch
import gensim
import os
import pathlib
import configparser



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

# gensim 词向量模型
model_path = MODEL_PATH + 'current_model.bin'
# 经过查看，常用的词组均在5w以内，10w词组以后我已经看不懂了，20w妥妥够用。没必要加载640w个词汇。
GENSIM_MODELS_WORD_LIMIT = 200000
MODEL = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, limit=GENSIM_MODELS_WORD_LIMIT)