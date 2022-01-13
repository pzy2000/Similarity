# -*- coding:utf-8 -*-
import pathlib
import os

root_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)  # 路径为 E:\PythonProject\demo
data_path = os.path.join(root_path, 'similarity\\data\\')
result_path = os.path.join(root_path, 'similarity\\result\\')
basedir2 = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)  # 路径为 E:\PythonProject\demo\similarity\src

# print(root_path)
# print(basedir2)




class Config():

    def __init__(self):
        # 数据元路径
        self.metadata_path = os.path.join(data_path, 'origin_metadata.csv')
        # 新增数据元路径
        # self.add_metadata_path = os.path.join(data_path, '某区目录-信息项数据2.0.xlsx')
        self.add_metadata_path = None
        # 新增数据元表单名称
        # self.add_metadata_sheet = None
        # 新增数据元表列名称
        # self.add_metadata_col = None
        # 模型表单表路径
        self.model_path = os.path.join(data_path, 'single_model.csv')
        # 模型表多表路径
        # self.model_path = os.path.join(data_path, 'multi_model.csv')
        # 目录表单表路径
        self.catalogue_path = os.path.join(data_path, 'single_catalogue.csv')
        # 目录表多表路径
        # self.catalogue_path = os.path.join(data_path, 'multi_catalogue.csv')
        # 已存在关联关系路径
        self.exist_asso_path = os.path.join(data_path, '人口库建设过程资料v0.1.xlsx')
        # 前k个候选数据元
        self.top_k = 10


        # ------------------------模型表相关------------------------
        # 模型表单表路径
        self.model_save_path = os.path.join(result_path, 'model_table\\single\\')
        # 模型表多表路径
        # self.model_save_path = os.path.join(result_path, 'model_table\\multi\\')
        # 模型表字段与数据元的关联关系json文件
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
        self.catalogue_save_path = os.path.join(result_path, 'catalogue_table\\multi\\')
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

