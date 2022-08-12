# -*- coding:utf-8 -*-
import pathlib
import os

root_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)  # 路径为 E:\PythonProject\demo
data_dir = os.path.join(root_path, 'similarity/data/')
result_dir = os.path.join(root_path, 'similarity/result/')
model_dir = os.path.join(root_path, 'similarity/model/')
basedir2 = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)  # 路径为 E:\PythonProject\demo\similarity\src





class Config():

    def __init__(self):
        # 数据元路径
        self.metadata_path = os.path.join(data_dir, 'metadata.csv')
        self.add_metadata_path = None
        self.model_path = os.path.join(data_dir, 'single_model.csv')
        self.catalogue_path = os.path.join(data_dir, 'single_catalogue.csv')
        self.exist_asso_path = os.path.join(data_dir, '人口库建设过程资料v0.1.xlsx')
        # 前k个候选数据元
        self.top_k = 3


        # ------------------------模型表相关------------------------
        # 模型表单表路径
        self.model_save_path = os.path.join(result_dir, 'model_table\\single\\')
        self.model_meta_name = 'model_meta.json'
        # 数据元与模型表字段的关联关系json文件
        self.meta_model_name = 'meta_model.json'
        # 模型表字段与多个候选数据元的关联关系json文件
        self.model_multimeta_name = 'model_meta_top' + str(self.top_k) + '.json'
        # 模型表字段与数据元的关联关系txt结果文件
        self.model_asso_name = 'model_asso_top' + str(self.top_k) + '.txt'


        # ------------------------目录表相关------------------------
        # 目录表单表路径
        # self.catalogue_save_path = os.path.join(result_path, 'catalogue_table\\single\\')
        # 目录表多表路径
        self.catalogue_save_path = os.path.join(result_dir, 'catalogue_table\\multi\\')
        # 目录表信息项与数据元关联关系的json文件
        self.catalogue_meta_name = 'catalogue_meta.json'
        # 数据元与目录表信息项关联关系的json文件
        self.meta_catalogue_name = 'meta_catalogue.json'
        # 目录表信息项与多个候选数据元关联关系的json文件
        self.catalogue_multimeta_name = 'catalogue_meta_top' + str(self.top_k) + '.json'
        # 目录表信息项与数据元的关联关系txt结果文件
        self.catalogue_asso_name = 'catalogue_asso_top' + str(self.top_k) + '.txt'
        # 最小置信下限，低于该阈值表明数据元中无合适字段进行关联
        self.min_confid = 0.2
        # 最大置信上限，高于该阈值表明关联关系结果可信
        self.max_confid = 0.7

