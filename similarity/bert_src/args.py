import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

file_path = os.path.dirname(os.path.dirname(__file__))
# print(file_path)


#模型目录
model_dir = os.path.join(file_path, 'albert_tiny_489k/')

#config文件
config_name = os.path.join(file_path, 'albert_tiny_489k/albert_config_tiny.json')
#ckpt文件名称
ckpt_name = os.path.join(model_dir, 'albert_model.ckpt')
#输出文件目录
# output_dir = os.path.join(file_path, 'result/')
output_dir = os.path.join(file_path, 'model/')
#vocab文件目录
vocab_file = os.path.join(file_path, 'albert_tiny_489k/vocab.txt')
# print(vocab_file)
#数据目录
data_dir = os.path.join(file_path, 'data/')

num_train_epochs = 10
batch_size = 128
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 128

# graph名字
graph_file = os.path.join(file_path, 'albert_lcqmc_checkpoints/graph')
# 初始模型
init_checkpoint = os.path.join(file_path, 'albert_tiny_489k/albert_model.ckpt')

do_train = False

do_predict = True