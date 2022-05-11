"""
机器学习算法有关代码，和业务逻辑无关
"""
import numpy as np
from typing import List
from similarity2.globals import MODEL, DEVICE
import torch


class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final = x.mul(1 / x_frobenins)
        # 替换nan值，可能会有0向量出现（某一方字符串为空）
        final = torch.where(torch.isnan(final), torch.full_like(final, 0), final)
        return final


_cs = CosineSimilarity().to(DEVICE)


def word2vec_avg_matrix(list_matrix: List[List[str]]) -> torch.Tensor:
    """
    :param list_matrix: 二维列表，每一行表示一个sample.
    :return: 词向量矩阵，(n_samples,n_features)

    将二维列表转化为平均词向量矩阵

    >>> X = [["我","想吃"],[],["皮卡丘"]]
    >>> y = word2vec_avg_matrix(X)
    >>> print(y.shape)
    (3,128)
    """
    sentences_matrix = []
    index = 0
    while index < len(list_matrix):
        words = list_matrix[index]
        if len(words) > 0:
            matrix = np.stack([MODEL[word] if word in MODEL else np.zeros(MODEL.vector_size) for word in words])
            feature = np.mean(matrix, axis=0)
        else:
            feature = np.zeros(MODEL.vector_size)
        sentences_matrix.append(feature)
        index += 1
    return torch.Tensor(np.stack(sentences_matrix)).to(DEVICE)


def cosine_similarity(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param X: (n_samples, n_features)
    :param y: (1,n_features)
    :return: (n_samples, 1)

    计算y和X的余弦相似度

    """
    X = X.to(DEVICE)
    y = y.to(DEVICE)

    return _cs(X, y)
