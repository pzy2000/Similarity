import collections
from queue import Queue
from threading import Thread
import tensorflow as tf
import  sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
# print(curPath)
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from similarity.bert_src.run_classifier import InputFeatures, InputExample, DataProcessor, convert_examples_to_features
from similarity import bert_src
import pandas as pd
from django.http import JsonResponse
import argparse

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import  multiprocessing
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

file_path = os.path.dirname(__file__)


#初始模型目录
args_model_dir = os.path.join(file_path, 'albert_tiny_489k/')

#config文件
args_config_name = os.path.join(file_path, 'albert_tiny_489k/albert_config_tiny.json')
#ckpt文件名称
args_ckpt_name = os.path.join(args_model_dir, 'albert_model.ckpt')
#输出文件目录
args_output_dir = os.path.join(file_path, 'model/')
#vocab文件目录
args_vocab_file = os.path.join(file_path, 'albert_tiny_489k/vocab.txt')
#数据目录
args_data_dir = os.path.join(file_path, 'data/')
more_sentences_path = os.path.join(file_path, 'data/pretrain.txt')
args_adddata_path = os.path.join(file_path, 'data/data.csv')

args_num_train_epochs = 10
args_batch_size = 128
args_learning_rate = 0.00005

# gpu使用率
args_gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
args_layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
args_max_seq_len = 128

# graph名字
args_graph_file = os.path.join(file_path, 'newmodel/graph')
# 初始模型
args_init_checkpoint = os.path.join(file_path, 'newmodel/model.ckpt')

args_do_train = False

args_do_predict = True

#记录当前训练服务状态
process_status = None
process_train = None
process_re_train = None
do_pretrain = False
do_train = False
do_retrain = False
# 配置模型参数
@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def config_model(request):
    global args_num_train_epochs
    global args_batch_size
    # gpu使用率
    global args_learning_rate
    global args_gpu_memory_fraction
    global args_max_seq_len
    parameter = request.data
    args_num_train_epochs = parameter['num_train_epochs']
    args_batch_size = parameter['batch_size']
    args_learning_rate = parameter['learning_rate']
    args_max_seq_len = parameter['max_seq_len']
    return Response({"code": 200, "msg": "修改成功！", "data": ""})

@csrf_exempt
def add_model_data(request):
    # 上传一个数据文件.csv
    if request.method == 'POST':
        # 解析上传文件
        myFile = request.FILES.get("files")
        # 保存文件
        filename = args_adddata_path
        f = open(filename, 'wb')
        for files in myFile.chunks():
            f.write(files)
        f.close()
        df = pd.read_csv(os.path.join(file_path, 'data/data.csv'), encoding='utf-8')
        df = df.sample(frac=1.0)  # 全部打乱
        cut_idx = int(round(0.1 * df.shape[0]))
        df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
        df_test.to_csv(os.path.join(file_path, 'data/test.csv'), index=False)
        df_train.to_csv(os.path.join(file_path, 'data/train.csv'), index=False)
        return JsonResponse({"code": 200, "msg": "上传文件成功！", "data": ""})
    return JsonResponse({"code": 404, "msg": "请使用POST方式请求！", "data": ""})

# 获取模型参数
@csrf_exempt
@api_view(http_method_names=['get'])  # 只允许get
@permission_classes((permissions.AllowAny,))
def get_model_config(request):
    return Response({"code": 200, "msg": "查看成功！", "data": {'num_train_epochs': args_num_train_epochs, 'batch_size': args_batch_size, 'learning_rate': args_learning_rate,

                     'gpu_memory_fraction': args_gpu_memory_fraction," args_max_seq_len" :  args_max_seq_len}})




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
        return JsonResponse({"code": 200, "msg": "上传文件成功！", "data": ""})
    return JsonResponse({"code": 404, "msg": "请使用POST方式请求！", "data": ""})

# 进行预训练
@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def do_pretrain(request):
    # 上传一个语料库文件.txt
    import subprocess

    process = subprocess.Popen("bash ./bert_src/create_pretrain_data.sh", cwd=file_path, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    global do_pretrain
    do_pretrain = True
    global process_status
    process_status = process

    return Response({"code": 200, "msg": "开始预训练！", "data": ""})

#获取预训练状态
@csrf_exempt
@api_view(http_method_names=['get'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def get_pretrain_state(request):
    global process_status
    if process_status == None:
        return HttpResponse({"code": 200, "msg": "没有预训练", "data": ""})
    process_status_now = process_status.poll()
    global do_pretrain
    if do_pretrain == False:
        return Response({"code": 200, "msg": "没有预训练", "data": ""})
    if process_status_now == None:
        return Response({"code": 200, "msg": "正在预训练", "data": ""})
    do_pretrain = None
    return Response({"code": 200, "msg": "完成预训练", "data": ""})

# 训练模型
@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def train_model(request):
    """
        需要train.tsv(训练集)，test.tsv(测试集)数据放入data下面
        训练数据格式：
        sent1,sent2,label
    """
    global do_train
    do_train = True
    global args_do_train
    args_do_train = True
    p = multiprocessing.Process(target=train_bert)
    p.start()
    global process_train
    process_train = p
    return Response({"code": 200, "msg": "模型训练开始！", "data": ""})

# 追加训练模型
@csrf_exempt
@api_view(http_method_names=['post'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def train_re_model(request):
    p = multiprocessing.Process(target=train_bert)
    p.start()
    global process_re_train
    global do_retrain
    process_re_train = p
    do_retrain = True
    return Response({"code": 200, "msg": "模型开始追加训练！", "data": ""})

#获取训练状态
@csrf_exempt
@api_view(http_method_names=['get'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def get_train_state(request):
    global process_train
    if process_train == None:
        return Response({"code": 200, "msg": "没有训练!", "data": ""})
    global do_train
    if do_train == False:
        return Response({"code": 200, "msg": "没有训练!", "data": ""})
    process_train_now = process_train.is_alive()
    if process_train_now == True:
        return Response({"code": 200, "msg": "正在训练!", "data": ""})
    process_train = None
    return Response({"code": 200, "msg": "完成训练!", "data": ""})

#获取追加训练状态
@csrf_exempt
@api_view(http_method_names=['get'])  # 只允许post
@permission_classes((permissions.AllowAny,))
def get_retrain_state(request):
    global process_re_train
    if  process_re_train == None:
        return Response({"code": 200, "msg": "没有追加训练!", "data": ""})
    global do_retrain
    if do_retrain == False:
        return Response({"code": 200, "msg": "没有追加训练!", "data": ""})
    process_re_train_now = process_re_train.is_alive()
    if process_re_train_now == True:
        return Response({"code": 200, "msg": "正在追加训练!", "data": ""})
    do_retrain = False
    return Response({"code": 200, "msg": "完成追加训练!", "data": ""})

def train_bert():
    sim = BertSim()
    global do_train
    global do_retrain
    global args_do_train
    args_do_train = True
    sim.set_mode(tf.estimator.ModeKeys.TRAIN)
    try:
        sim.train()
    except ValueError:
        do_train = False
        do_retrain = False
        args_do_train = False
        return
    do_train = False
    do_retrain = False
    args_do_train = False

def train_bert_cmd(batch_size,learningrate,seqlen,epochs):
    global args_num_train_epochs
    global args_batch_size
    # gpu使用率
    global args_learning_rate
    global args_gpu_memory_fraction
    global args_max_seq_len
    if epochs !=None:
        args_max_seq_len=epochs
    if batch_size !=None:
        args_batch_size=batch_size
    if learningrate !=None:
        args_learning_rate=learningrate
    if seqlen != None:
        args_max_seq_len=seqlen
    df = pd.read_csv(os.path.join(file_path, 'data/data.csv'), encoding='utf-8')
    df = df.sample(frac=1.0)  # 全部打乱
    cut_idx = int(round(0.1 * df.shape[0]))
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    df_test.to_csv(os.path.join(file_path, 'data/test.csv'), index=False)
    df_train.to_csv(os.path.join(file_path, 'data/train.csv'), index=False)
    print("训练开始")
    sim = BertSim()
    global do_train
    global do_retrain
    global args_do_train
    args_do_train = True
    sim.set_mode(tf.estimator.ModeKeys.TRAIN)
    try:
        sim.train()
    except ValueError:
        do_train = False
        do_retrain = False
        args_do_train = False
        return
    do_train = False
    do_retrain = False
    args_do_train = False
    print("训练结束")

class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        if args_do_train!=True:
          return [];
        file_path = os.path.join(data_dir, 'train.csv')
        train_df = pd.read_csv(file_path, encoding='utf-8')
        train_data = []
        for index, train in enumerate(train_df.values):
            guid = 'train-%d' % index
            text_a = bert_src.tokenization.convert_to_unicode(str(train[0]))
            text_b = bert_src.tokenization.convert_to_unicode(str(train[1]))
            label = str(train[2])
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return train_data

    def get_dev_examples(self, data_dir):
        if args_do_train != True:
            return []
        file_path = os.path.join(data_dir, 'dev.csv')
        dev_df = pd.read_csv(file_path, encoding='utf-8')
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'test-%d' % index
            text_a = bert_src.tokenization.convert_to_unicode(str(dev[0]))
            text_b = bert_src.tokenization.convert_to_unicode(str(dev[1]))
            label = str(dev[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return dev_data

    def get_test_examples(self, data_dir):
        if args_do_train != True:
            return []
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path, encoding='utf-8')
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = bert_src.tokenization.convert_to_unicode(str(test[0]))
            text_b = bert_src.tokenization.convert_to_unicode(str(test[1]))
            label = str(test[2])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return test_data

    @staticmethod
    def get_sentence_examples(questions):
        for index, data in enumerate(questions):
            guid = 'test-%d' % index
            text_a = bert_src.tokenization.convert_to_unicode(str(data[0]))
            text_b = bert_src.tokenization.convert_to_unicode(str(data[1]))
            label = str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        return ['0', '1']


class BertSim():

    def __init__(self, batch_size=args_batch_size):

        self.mode = None
        self.max_seq_len = args_max_seq_len
        self.tokenizer = bert_src.tokenization.FullTokenizer(vocab_file=args_vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = None
        self.processor = SimProcessor()    # 加载训练、测试数据class
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    def set_mode(self, mode):
        self.mode = mode
        self.estimator = self.get_estimator()
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.input_queue = Queue(maxsize=1)
            self.output_queue = Queue(maxsize=1)
            self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)#daemon守护进程
            self.predict_thread.start()

    def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        model = bert_src.modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)


        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.compat.v1.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.compat.v1.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.compat.v1.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)

    @staticmethod
    def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps,
                         use_one_hot_embeddings):
        """Returns `model_fn` closurimport_tfe for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            from tensorflow.python.estimator.model_fn import EstimatorSpec

            # # tf.compat.v1.logging.info("*** Features ***")

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities) = BertSim.create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)

            tvars = tf.compat.v1.trainable_variables()
            initialized_variable_names = {}

            if init_checkpoint:
                (assignment_map, initialized_variable_names) = bert_src.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

            # # tf.compat.v1.logging.info("**** Trainable Variables ****")

            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = bert_src.optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

                output_spec = EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, logits):
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    accuracy = tf.metrics.accuracy(label_ids, predictions)
                    auc = tf.metrics.auc(label_ids, predictions)
                    loss = tf.metrics.mean(per_example_loss)
                    return {
                        "eval_accuracy": accuracy,
                        "eval_auc": auc,
                        "eval_loss": loss,
                    }

                eval_metrics = metric_fn(per_example_loss, label_ids, logits)
                output_spec = EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=eval_metrics)
            else:
                output_spec = EstimatorSpec(mode=mode, predictions=probabilities)

            return output_spec

        return model_fn

    def get_estimator(self):

        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig

        bert_config = bert_src.modeling.BertConfig.from_json_file(args_config_name)
        label_list = self.processor.get_labels()
        train_examples = self.processor.get_train_examples(args_data_dir)
        num_train_steps = int(len(train_examples) / self.batch_size * args_num_train_epochs)
        num_warmup_steps = int(num_train_steps * 0.1)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            init_checkpoint = args_init_checkpoint
        else:
            init_checkpoint = args_output_dir   # 预测模式下加载

        model_fn = self.model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=init_checkpoint,
            learning_rate=args_learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_one_hot_embeddings=False)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = args_gpu_memory_fraction
        config.log_device_placement = False

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config), model_dir=args_output_dir,
                         params={'batch_size': self.batch_size})

    def predict_from_queue(self):
        for i in self.estimator.predict(input_fn=self.queue_predict_input_fn, yield_single_examples=False):
            self.output_queue.put(i)

    def queue_predict_input_fn(self):
        return (tf.data.Dataset.from_generator(
            self.generate_from_queue,
            output_types={
                'input_ids': tf.int32,
                'input_mask': tf.int32,
                'segment_ids': tf.int32,
                'label_ids': tf.int32},
            output_shapes={
                'input_ids': (None, self.max_seq_len),
                'input_mask': (None, self.max_seq_len),
                'segment_ids': (None, self.max_seq_len),
                'label_ids': (1,)}).prefetch(10))

    def convert_examples_to_features(self, examples, label_list, max_seq_len, tokenizer):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        for (ex_index, example) in enumerate(examples):
            label_map = {}
            for (i, label) in enumerate(label_list):
                label_map[label] = i

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_len - 2:
                    tokens_a = tokens_a[0:(max_seq_len - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            label_id = label_map[example.label]

            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id)

            yield feature

    def generate_from_queue(self):
        while True:
            predict_examples = self.processor.get_sentence_examples(self.input_queue.get())
            features = list(self.convert_examples_to_features(predict_examples, self.processor.get_labels(),
                                                              args_max_seq_len, self.tokenizer))
            yield {
                'input_ids': [f.input_ids for f in features],
                'input_mask': [f.input_mask for f in features],
                'segment_ids': [f.segment_ids for f in features],
                'label_ids': [f.label_id for f in features]
            }

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_single_example(self, ex_index, example, label_list, max_seq_len, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_len - 2:
                tokens_a = tokens_a[0:(max_seq_len - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        label_id = label_map[example.label]

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id)
        return feature

    def file_based_convert_examples_to_features(self, examples, label_list, max_seq_len, tokenizer, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):

            feature = self.convert_single_example(ex_index, example, label_list,
                                                  max_seq_len, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

    @staticmethod
    def file_based_input_fn_builder(input_file, seq_length,
                                    is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

            return d

        return input_fn

    def train(self):
        if self.mode is None:
            raise ValueError("Please set the 'mode' parameter")

        bert_config = bert_src.modeling.BertConfig.from_json_file(args_config_name)

        if args_max_seq_len > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (args_max_seq_len, bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(args_output_dir)

        label_list = self.processor.get_labels()

        train_examples = self.processor.get_train_examples(args_data_dir)
        num_train_steps = int(len(train_examples) / args_batch_size * args_num_train_epochs)

        estimator = self.get_estimator()

        train_file = os.path.join(args_output_dir, "train.tf_record")
        self.file_based_convert_examples_to_features(train_examples, label_list, args_max_seq_len, self.tokenizer,
                                                     train_file)
        # # tf.compat.v1.logging.info("***** Running training *****")
        # # tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        # # tf.compat.v1.logging.info("  Batch size = %d", args_batch_size)
        # # tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = self.file_based_input_fn_builder(input_file=train_file, seq_length=args_max_seq_len,
                                                          is_training=True,
                                                          drop_remainder=True)


        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    def eval(self):
        if self.mode is None:
            raise ValueError("Please set the 'mode' parameter")
        eval_examples = self.processor.get_dev_examples(args_data_dir)
        eval_file = os.path.join(args_output_dir, "eval.tf_record")
        label_list = self.processor.get_labels()
        self.file_based_convert_examples_to_features(
            eval_examples, label_list, args_max_seq_len, self.tokenizer, eval_file)

        # # tf.compat.v1.logging.info("***** Running evaluation *****")
        # # tf.compat.v1.logging.info("  Num examples = %d", len(eval_examples))
        # # tf.compat.v1.logging.info("  Batch size = %d", self.batch_size)

        eval_input_fn = self.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=args_max_seq_len,
            is_training=False,
            drop_remainder=False)

        estimator = self.get_estimator()
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

        output_eval_file = os.path.join(args_output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    def predict(self, sentence1, sentence2):
        if self.mode is None:
            raise ValueError("Please set the 'mode' parameter")
        self.input_queue.put([(sentence1, sentence2)])
        prediction = self.output_queue.get()
        return prediction



def input_fn_builder(bertSim,sentences):
    def predict_input_fn():
        return (tf.data.Dataset.from_generator(
            generate_from_input,
            output_types={
                'input_ids': tf.int32,
                'input_mask': tf.int32,
                'segment_ids': tf.int32,
                'label_ids': tf.int32},
            output_shapes={
                'input_ids': (None, bertSim.max_seq_len),
                'input_mask': (None, bertSim.max_seq_len),
                'segment_ids': (None, bertSim.max_seq_len),
                'label_ids': (1,)}).prefetch(10))

    def generate_from_input():
        processor = bertSim.processor
        predict_examples = processor.get_sentence_examples(sentences)
        features = convert_examples_to_features(predict_examples, processor.get_labels(), args_max_seq_len,
                                                bertSim.tokenizer)
        yield {
            'input_ids': [f.input_ids for f in features],
            'input_mask': [f.input_mask for f in features],
            'segment_ids': [f.segment_ids for f in features],
            'label_ids': [f.label_id for f in features]
        }

    return predict_input_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser("you should add those parameter")
    parser.add_argument('-d', dest='d', type=str, help='Operations on the model')
    parser.add_argument('-batch', dest='batch_size', type=int, help='The batch size for the model')
    parser.add_argument('-learningrate', dest='learnrate', type=int, help='The learning rate in the model')
    parser.add_argument('-seqlen', dest='seqlen', type=int, help='Max sequence length of the model')
    parser.add_argument('-epochs', dest='epochs', type=int, help='the number of the epoch for model trainning ')
    args = parser.parse_args()
    print(args.batch_size,args.learnrate,args.seqlen,args.epochs)
    if args.d:
        if args.d == 'train':
            train_bert_cmd(args.batch_size,args.learnrate,args.seqlen,args.epochs)
        elif args.d == 'retrain':
            train_bert_cmd(args.batch_size,args.learnrate,args.seqlen,args.epochs)
        else:
            print('参数错误')
    else:
        print("缺少参数")
