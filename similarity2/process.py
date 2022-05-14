"""
数据处理相关代码，和业务逻辑相关
"""
from typing import List, Union, Dict, Any, Tuple
from similarity2.algorithm import word2vec_avg_matrix, cosine_similarity
from similarity2.globals import DEVICE
import torch
import jieba


def match_str2matrix(data: Union[List[str], str], sep='^', word2vec=word2vec_avg_matrix) -> torch.Tensor:
    """
    :param data: 字符串列表或字符串。通过分隔符分割的数据
    :param sep: 字符。分割符，默认为"^"
    :param word2vec: 函数。词向量计算方法，默认为word2vec_avg_matrix
    :return: (n_items, n_samples, n_features)

    n_items表示match_str分割后的数量 \n

    example: \n
    假设有 30 个 match_str \n
    假设每个match_str可以分成5个，例如"迪卢克^刻晴^七七^琴^莫娜" \n
    则通过gensim模型转化为词向量后，返回的数据shape为 (5,30,n_features) \n

    >>> from similarity2.database import Database
    >>> db = Database.get_common_database()
    >>> r = db.filter(business_type="catalog_item").values_list("match_str",flat=True)
    >>> matrix = match_str2matrix(r)
    >>> matrix.shape
    (5,8,127)
    """
    if type(data) == str:
        data = [data]

    data = \
        torch.stack([
            # stack: (n_items, n_features) -> (n_items,n_samples,n_features)
            # cat: (1,n_features) -> (n_items,n_features)
            torch.cat([
                word2vec_avg_matrix(
                    # 二维列表
                    [
                        jieba.lcut(item, cut_all=True, HMM=True)
                    ]
                )
                # 对每个item进行分词，并产生一个词向量
                for item in match_str.split("^")
            ], dim=0)
            # 每个match_str产生n_items个词向量
            for match_str in data
        ], dim=1)

    data = data.to(DEVICE)
    return data


def vector_match(X: torch.Tensor, y: torch.Tensor, weight: List[float], k: int) -> Tuple[List[int], List[float]]:
    """
    :param X: (n_items, n_samples, n_features) 数据集match_str词向量
    :param y: (n_items, 1, n_features) 待匹配match_str的词向量
    :param weight: 权重列表
    :param k: 返回的匹配个数
    :return: 元组，内容为：(索引列表，匹配值列表)

    """
    n_items, _, _ = X.shape

    # (可以优化成矩阵直接相乘，性能影响不大，先for循环着吧)
    # (n_items,n_samples,1)
    sim_value: List[torch.Tensor] = [
        cosine_similarity(X[i], y[i]) * weight[i]
        for i in range(n_items)
    ]


    # 按照n_items维度求和
    # (n_samples, 1)
    sim_value: torch.Tensor = torch.sum(torch.stack(sim_value), dim=0)

    # 替换nan值，可能会有0向量出现（某一方字符串为空）
    sim_value: torch.Tensor = torch.where(torch.isnan(sim_value), torch.full_like(sim_value, 0), sim_value)

    # 按照n_samples维度排序，选择topk
    value, index = torch.topk(sim_value, k, dim=0, largest=True, sorted=True)

    # 转为列表，便于数据处理
    value = value.numpy().ravel().tolist()
    index = index.numpy().ravel().tolist()


    # 对value进行限制
    value = [1 if v > 1 else (0 if v < 0 else v) for v in value]

    return index, value
