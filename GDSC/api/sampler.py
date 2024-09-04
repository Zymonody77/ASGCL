import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold
import torch


class MultipleTargetSampler(object):
    def __init__(self, adj: np.ndarray, null_mask: np.ndarray, indexes: np.ndarray, cv_num=5, shuffle=True,
                 random_state=22):
        super(MultipleTargetSampler, self).__init__()
        self.adj = adj
        self.null_mask = null_mask
        self.indexes = indexes
        self.kfold = KFold(n_splits=cv_num, shuffle=shuffle, random_state=random_state)

    def sample_train_test_data(self, pos_train_index: np.ndarray, pos_test_index: np.ndarray):
        n_target = self.indexes.shape[0]
        target_response = self.adj[:, self.indexes].reshape((-1, n_target))
        train_data = self.adj.copy()
        train_data[:, self.indexes] = 0
        target_pos_value = sp.coo_matrix(target_response)
        target_train_data = sp.coo_matrix((target_pos_value.data[pos_train_index],
                                           (target_pos_value.row[pos_train_index],
                                            target_pos_value.col[pos_train_index])),
                                          shape=target_response.shape).toarray()
        target_test_data = sp.coo_matrix((target_pos_value.data[pos_test_index],
                                          (target_pos_value.row[pos_test_index],
                                           target_pos_value.col[pos_test_index])),
                                         shape=target_response.shape).toarray()
        test_data = np.zeros(self.adj.shape, dtype=np.float32)
        for i, value in enumerate(self.indexes):
            train_data[:, value] = target_train_data[:, i]
            test_data[:, value] = target_test_data[:, i]
        return train_data, test_data

    def sample_train_test_mask(self, test_number: int):
        neg_value = np.ones(self.adj.shape, dtype=np.float32) - self.adj - self.null_mask
        target_neg = neg_value[:, self.indexes].reshape((-1, self.indexes.shape[0]))
        sp_target_neg = sp.coo_matrix(target_neg)
        ids = np.arange(sp_target_neg.data.shape[0])
        target_neg_test_index = np.random.choice(ids, test_number, replace=False)
        target_neg_test_mask = sp.coo_matrix((sp_target_neg.data[target_neg_test_index],
                                              (sp_target_neg.row[target_neg_test_index],
                                               sp_target_neg.col[target_neg_test_index])),
                                             shape=target_neg.shape).toarray()
        neg_test_mask = np.zeros(self.adj.shape, dtype=np.float32)
        for i, value in enumerate(self.indexes):
            neg_test_mask[:, value] = target_neg_test_mask[:, i]
        neg_train_mask = neg_value - neg_test_mask
        return neg_train_mask, neg_test_mask

    def __call__(self, dtype, device):
        target_adj = self.adj[:, self.indexes].reshape((-1, self.indexes.shape[0]))
        sp_target_adj = sp.coo_matrix(target_adj)
        for train_index, test_index in self.kfold.split(range(sp_target_adj.data.shape[0])):
            test_index = np.array(test_index)
            train_data, test_data = self.sample_train_test_data(pos_train_index=train_index, pos_test_index=test_index)
            train_mask, test_mask = self.sample_train_test_mask(test_number=test_index.shape[0])
            train_mask = (train_data + train_mask).astype(np.bool_)
            test_mask = (test_mask + test_data).astype(np.bool_)
            train_data = torch.from_numpy(train_data).to(dtype=dtype, device=device)
            test_data = torch.from_numpy(test_data).to(dtype=dtype, device=device)
            train_mask = torch.from_numpy(train_mask).to(dtype=torch.bool, device=device)
            test_mask = torch.from_numpy(test_mask).to(dtype=torch.bool, device=device)
            yield train_data, test_data, train_mask, test_mask