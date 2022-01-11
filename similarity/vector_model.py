# coding=utf-8
import multiprocessing
import os

import gensim
import jieba
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from gensim.models.word2vec import Word2Vec, LineSentence
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

window = 5
min_count = 1
dim = 128

origin_corpus_path = os.getcwd() + '/similarity/corpus/gov.txt'
seg_corpus_path = os.getcwd() + '/similarity/corpus/corpus.txt'
more_sentences_path = os.getcwd() + '/similarity/corpus/more_sentences.txt'
# 默认模型
model_dir = os.getcwd() + '/similarity/model/'
model_path = model_dir + 'baike_26g_news_13g_novel_229g.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


# 配置模型参数
@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def config_model(request):
    global window
    global dim
    global min_count
    parameter = request.data
    window = parameter['window']
    dim = parameter['dim']
    min_count = parameter['min_count']
    return HttpResponse("模型配置成功！")


# 获取模型参数
@csrf_exempt
@api_view(http_method_names=['get'])  # 只允许get
@permission_classes((permissions.AllowAny,))
def get_model_config(request):
    return Response({'window': window, 'min_count': min_count, 'dim': dim})


# 新增语料库
@csrf_exempt
def add_corpus(request):
    # 上传一个语料库文件.txt
    if request.method == 'POST':
        myFile = request.FILES.get("corpus")
        f = open(more_sentences_path, 'wb')
        for files in myFile.chunks():
            f.write(files)
        f.close()
        return HttpResponse("上传文件成功！")
    return HttpResponse("请使用POST请求上传文件")

# 训练模型
@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def train_model(request):
    f = open(origin_corpus_path, encoding='utf-8')
    line = f.readline()
    # stopwords = [line.strip() for line in open(r'./stopwords/cn_stopwords.txt', 'r', encoding='utf-8')]
    seg_list = []
    count = 0
    while line:
        if line != '\n' and line != '':
            segment = jieba.lcut(line, cut_all=True, HMM=True)
            print(count)
            count += 1
            for s in segment:
                for i in range(0, len(s)):
                    if is_all_chinese(s[i]):
                        seg_list.append(s[i])
        line = f.readline()
    f2 = open(seg_corpus_path, 'w', encoding='utf-8')
    for s in seg_list:
        f2.write(s + ' ')
    tmp_model = Word2Vec(LineSentence(seg_corpus_path), window=window, min_count=min_count, vector_size=dim,
                         workers=multiprocessing.cpu_count())
    tmp_model.save(model_dir + "word2vec.model")
    tmp_model.wv.save_word2vec_format(model_dir + "word2vec.vector", binary=True)
    return HttpResponse("模型训练完毕！")


# 追加训练模型
@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def retrain_model(request):
    tmp_model = gensim.models.Word2Vec.load(model_dir + 'word2vec.model')
    f = open(more_sentences_path, encoding='utf-8')
    line = f.readline()
    # stopwords = [line.strip() for line in open(r'./stopwords/cn_stopwords.txt', 'r', encoding='utf-8')]
    seg_list = []
    count = 0
    while line:
        if line != '\n' and line != '':
            segment = jieba.lcut(line, cut_all=True, HMM=True)
            print(count)
            count += 1
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
    return HttpResponse("模型追加训练完成！")


# 判断是否为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True
