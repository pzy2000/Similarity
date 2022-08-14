# -*- coding:utf-8 -*-
import pathlib
import os

root_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)  # 路径为 E:\PythonProject\Similarity
data_dir = os.path.join(root_path, 'similarity/data/')
result_dir = os.path.join(root_path, 'similarity/result/')
model_dir = os.path.join(root_path, 'similarity/model/')

# print(root_path)
