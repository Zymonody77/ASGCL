import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef, average_precision_score


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def gaussian_normalization(x: torch.Tensor):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    return torch.div(torch.sub(x, mean.view([1, -1])), std.view([1, -1]))


def euclidean_distance(x: torch.Tensor):
    x = torch.mm(x, x.t())
    diag = torch.diag(x)
    diag = torch.add(diag.view([-1, 1]), diag.view([1, -1]))
    x = torch.sub(diag, torch.mul(x, 2.))
    return torch.sqrt(x)


def exponent_similarity(dist: torch.Tensor, mu: float):
    denominate = torch.square(torch.tensor(mu, dtype=dist.dtype, device=dist.device))
    dist = torch.div(dist, torch.mul(2, denominate))
    dist = torch.exp(-dist)
    return dist


def jaccard_coef(tensor: torch.Tensor):
    """
    :param tensor: an torch tensor, 2D
    :return: jaccard coefficient
    """
    assert torch.all(tensor.le(1)) and torch.all(tensor.ge(0)), "Value must be 0 or 1"
    size = tensor.size()
    tensor_3d = torch.flatten(tensor).repeat([size[0]]).view([size[0], size[0], size[1]])
    ones = torch.ones_like(tensor_3d)
    zeros = torch.zeros_like(tensor_3d)
    tensor_3d = torch.add(tensor_3d, tensor.view([size[0], 1, size[1]]))
    intersection = torch.where(tensor_3d.eq(2), ones, zeros)
    union = torch.where(tensor_3d.eq(2), ones, tensor_3d)
    intersection = torch.sum(intersection, dim=2)
    union = torch.sum(union, dim=2)
    union = torch.where(union.eq(0), torch.add(union, 0.1), union)
    eye = torch.eye(union.size()[0], dtype=tensor.dtype, device=tensor.device)
    jaccard = torch.div(intersection, union)
    jaccard = torch.where(jaccard.eq(0), eye, jaccard)
    return jaccard


def to_data_frame(data: torch.Tensor or np.ndarray):
    """
    Convert torch.Tensor to pd.DataFrame to save
    :param data: torch tensor
    :return: pd.DataFrame
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = pd.DataFrame(data.reshape((1, -1)))
    return data


def roc_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: AUC score
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    aucs = roc_auc_score(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
    return aucs


def f1_res(true_data, predict_data):
    f1 = f1_score(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy().round().astype(int))
    return f1


def acc_score(true_data, predict_data):
    acc = accuracy_score(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy().round().astype(int))
    return acc


def mcc_score(true_data, predict_data):
    mcc = matthews_corrcoef(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy().round().astype(int))
    return mcc


def ap_score(true_data, predict_data):
    ap = average_precision_score(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
    return ap


def common_data_index(data_for_index: np.ndarray, data_for_cmp: np.ndarray):
    """
    :param data_for_index: data for index, numpy array
    :param data_for_cmp: data for compare, numpy array
    :return: index of common data in data for index
    """
    index = np.array([np.where(x in data_for_cmp, 1, 0) for x in data_for_index])
    index = np.where(index == 1)[0]
    return index


def calculate_gene_exponent_similarity(x: torch.Tensor, mu: float):
    """
    Calculate the gene exponent similarity
    :param x: gene feature
    :param mu: scale parameter
    :return: cell line gene exponent similarity
    """
    x = gaussian_normalization(x=x)
    x = euclidean_distance(x=x)
    return exponent_similarity(dist=x, mu=mu)
