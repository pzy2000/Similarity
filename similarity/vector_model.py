import argparse
import multiprocessing

import gensim
import jieba
from gensim.models.word2vec import Word2Vec, LineSentence

window = 5
min_count = 1
dim = 128
model_dir = './model/'
origin_corpus_path = './corpus/gov.txt'
seg_corpus_path = './corpus/corpus.txt'
more_sentences_path = './corpus/more_sentences.txt'


# 训练模型
def train_model(i_window, i_min_count, i_dim):
    f = open(origin_corpus_path, encoding='utf-8')
    line = f.readline()
    seg_list = []
    print("------------------------训练模型--------------------------")
    while line:
        if line not in ('\n', ''):
            segment = jieba.lcut(line, cut_all=True, HMM=True)
            for s in segment:
                for i in range(0, len(s)):
                    if is_all_chinese(s[i]):
                        seg_list.append(s[i])
        line = f.readline()
    f2 = open(seg_corpus_path, 'w', encoding='utf-8')
    for s in seg_list:
        f2.write(s + ' ')
    if not i_window:
        i_window = window
    if not i_min_count:
        i_min_count = min_count
    if not i_dim:
        i_dim = dim
    tmp_model = Word2Vec(LineSentence(seg_corpus_path), window=i_window, min_count=i_min_count, vector_size=i_dim,
                         workers=multiprocessing.cpu_count())
    tmp_model.save(model_dir + "word2vec.model")
    tmp_model.wv.save_word2vec_format(model_dir + "word2vec.vector", binary=True)
    print("-----------------------训练模型结束------------------------")


# 追加训练模型
def retrain_model():
    tmp_model = gensim.models.Word2Vec.load(model_dir + 'word2vec.model')
    f = open(more_sentences_path, encoding='utf-8')
    line = f.readline()
    seg_list = []
    print("-----------------------开始追加训练模型------------------------")
    while line:
        if line not in ('\n', ''):
            segment = jieba.lcut(line, cut_all=True, HMM=True)
            for s in segment:
                for i in range(0, len(s)):
                    if is_all_chinese(s[i]):
                        seg_list.append(s[i])
        line = f.readline()
    more_sentences = [seg_list]
    tmp_model.build_vocab(more_sentences, update=True)
    tmp_model.train(more_sentences, total_examples=tmp_model.corpus_count, epochs=tmp_model.epochs)
    tmp_model.save(model_dir + "word2vec.model")
    tmp_model.wv.save_word2vec_format(model_dir + "word2vec.vector", binary=True)
    print("-----------------------追加训练模型结束------------------------")


# 判断是否为中文
def is_all_chinese(strs):
    return all('\u4e00' <= _char <= '\u9fa5' for _char in strs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("you should add those parameter")
    parser.add_argument('-d', dest='d', type=str, help='Operations on the model')
    parser.add_argument('-window', dest='window', type=int, help='The size of window for the model')
    parser.add_argument('-min_count', dest='min_count', type=int, help='The lowest word frequency counted in the model')
    parser.add_argument('-dim', dest='dim', type=int, help='Vector size')
    args = parser.parse_args()
    if args.d:
        if args.d == 'train':
            train_model(window, min_count, dim)
        elif args.d == 'retrain':
            retrain_model()
        else:
            print('参数错误')
    else:
        print("缺少参数")
