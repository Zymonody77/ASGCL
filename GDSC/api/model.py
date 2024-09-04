import sys
import os
from abc import ABC
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun
import torch.nn.functional as F
from GDSC.api.utils import *
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(__file__))

EPS = 1e-15


class FeatureGraphConvolution(nn.Module, ABC):
    def __init__(self, sim: torch.Tensor, knn: int, in_dim: int, embed_dim: int, act=fun.relu, bias=False):
        super(FeatureGraphConvolution, self).__init__()
        self.sim = sim
        self.knn = knn
        self.diffuse_lm = nn.Linear(in_dim, embed_dim, bias=bias)
        self.act = act
        self.laplace = self.__calculate_laplace()

    @staticmethod
    def k_near_graph(sim: torch.Tensor, k: int):
        """
        Calculate the k near graph as feature space adjacency.
        :param sim: similarity matrix, torch.Tensor
        :param k: k, int
        :return: weighted adjacency matrix
        """
        threshold = torch.min(torch.topk(sim, k=k, dim=1).values, dim=1).values.view([-1, 1])
        sim = torch.where(sim.ge(threshold), sim, torch.zeros_like(sim))
        return sim

    @staticmethod
    def diffuse_laplace(adj: torch.Tensor):
        d_x = torch.diag(torch.pow(torch.sum(adj, dim=1), -0.5))
        d_y = torch.diag(torch.pow(torch.sum(adj, dim=0), -0.5))
        adj = torch.mm(torch.mm(d_x, adj), d_y)
        return adj

    def __calculate_laplace(self):
        adj = FeatureGraphConvolution.k_near_graph(sim=self.sim, k=self.knn)
        diffuse_x_laplace = FeatureGraphConvolution.diffuse_laplace(adj=adj)
        return diffuse_x_laplace

    def forward(self, x: torch.Tensor):
        out = self.act(self.diffuse_lm(torch.mm(self.laplace, x)))
        return out


class TopologicalGraphConvolution(nn.Module, ABC):
    def __init__(self, adj: torch.Tensor, in_dim: int, embed_dim: int, act=fun.relu, bias=False):
        super(TopologicalGraphConvolution, self).__init__()
        self.laplace = TopologicalGraphConvolution.diffuse_laplace(adj=adj)
        self.act = act
        self.lm = nn.Linear(in_dim, embed_dim, bias=bias)

    @staticmethod
    def diffuse_laplace(adj: torch.Tensor):
        d_x = torch.diag(torch.pow(torch.add(torch.sum(adj, dim=1), 1), -0.5))
        d_y = torch.diag(torch.pow(torch.add(torch.sum(adj, dim=0), 1), -0.5))
        adj = torch.mm(torch.mm(d_x, adj), d_y)
        return adj

    def forward(self, opposite_x: torch.Tensor, opposite_feature_x: torch.Tensor):
        collected_opposite_x = self.act(self.lm(torch.mm(self.laplace, opposite_x)))
        collected_feature_x = torch.mm(self.laplace, opposite_feature_x)
        return [collected_opposite_x, collected_feature_x]


class SelfFeatureExtractor(nn.Module, ABC):
    def __init__(self, adj: torch.Tensor, in_dim: int, embed_dim: int, bias=False, act=fun.relu):
        super(SelfFeatureExtractor, self).__init__()
        self.laplace = SelfFeatureExtractor.self_laplace(adj=adj)
        self.lm = nn.Linear(in_dim, embed_dim, bias=bias)
        self.act = act

    @staticmethod
    def self_laplace(adj: torch.Tensor):
        d = torch.pow(torch.add(torch.sum(adj, dim=1), 1), -1)
        d = torch.diag(torch.add(d, 1))
        return d

    def forward(self, x: torch.Tensor):
        x = self.act(self.lm(torch.mm(self.laplace, x)))
        return x


class CorrelationDecoder(nn.Module, ABC):
    def __init__(self, embed_dim: int, kernel_dim: int, alpha: float):
        super(CorrelationDecoder, self).__init__()
        self.lm_x = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.lm_y = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.alpha = alpha

    @staticmethod
    def corr_x_y(x: torch.Tensor, y: torch.Tensor):
        assert x.size()[1] == y.size()[1], "Different size!"
        x = torch.sub(x, torch.mean(x, dim=1).view([-1, 1]))
        y = torch.sub(y, torch.mean(y, dim=1).view([-1, 1]))
        lxy = torch.mm(x, y.t())
        lxx = torch.diag(torch.mm(x, x.t()))
        lyy = torch.diag(torch.mm(y, y.t()))
        std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
        corr = torch.div(lxy, std_x_y)
        return corr

    @staticmethod
    def scale_sigmoid_activation_function(x: torch.Tensor, alpha: int or float):
        assert torch.all(x.ge(-1)) and torch.all(x.le(1)), "Out of range!"
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
        x = torch.sigmoid(torch.mul(alpha, x))
        max_value = torch.sigmoid(alpha)
        min_value = torch.sigmoid(-alpha)
        output = torch.div(torch.sub(x, min_value), torch.sub(max_value, min_value))
        return output

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.lm_x(x)
        y = self.lm_y(y)
        out = CorrelationDecoder.corr_x_y(x=x, y=y)
        out = CorrelationDecoder.scale_sigmoid_activation_function(x=out, alpha=self.alpha)
        return out


class GraphConvolutionEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.PReLU(), base_model=GCNConv, k: int = 2,
                 skip=False):
        super(GraphConvolutionEncoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class BiSpaceGraphConvolution(nn.Module, ABC):
    def __init__(self, adj: torch.Tensor, x_sim: torch.Tensor, y_sim: torch.Tensor, mask: torch.Tensor,
                 embed_dim: int, kernel_dim: int, pos_edge_idx, neg_edge_idx, eta, **kwargs):
        super(BiSpaceGraphConvolution, self).__init__()
        self.adj = adj
        self.x_sim = x_sim
        self.x = BiSpaceGraphConvolution.gaussian_normalization(x=x_sim)
        self.y_sim = y_sim
        self.y = y_sim
        self.mask = mask
        self.embed_dim = embed_dim
        self.kernel_dim = kernel_dim
        self.act = kwargs.get("act", fun.relu)
        self.alpha = kwargs.get("alpha", 5.74)
        self.beta = kwargs.get("beta", 1.75)
        self.x_knn = kwargs.get("x_knn", x_sim.size()[1])
        self.y_knn = kwargs.get("y_knn", y_sim.size()[1])
        self.self_space = self._add_self_space()
        self.topo_space = self._add_topo_space()
        self.feature_space = self._add_feature_space()
        self.predict_layer = self._add_predict_layer()
        self.pos_edge_idx = pos_edge_idx
        self.neg_edge_idx = neg_edge_idx
        self.gcn_encoder = GraphConvolutionEncoder(224, 224)
        num_hidden = 224
        num_proj_hidden = 1190
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.eta = eta

    @staticmethod
    def gaussian_normalization(x: torch.Tensor):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = torch.div(torch.sub(x, mean.view([1, -1])), std.view([1, -1]))
        return x

    @staticmethod
    def full_kernel(similarity: torch.Tensor):
        mask_diag = torch.sub(similarity, torch.diag(torch.diag(similarity)))
        mask_diag = torch.div(mask_diag, torch.mul(2, torch.sum(mask_diag, dim=1).view([-1, 1])))
        mask_diag = torch.add(mask_diag, torch.mul(0.5, torch.eye(similarity.size()[0],
                                                                  dtype=similarity.dtype, device=similarity.device)))
        return mask_diag

    def _add_self_space(self):
        x = SelfFeatureExtractor(adj=self.adj, in_dim=self.x.size()[1], embed_dim=self.embed_dim)
        y = SelfFeatureExtractor(adj=self.adj.t(), in_dim=self.y.size()[1], embed_dim=self.embed_dim)
        return nn.ModuleList([x, y])

    def _add_topo_space(self):
        x = TopologicalGraphConvolution(adj=self.adj, in_dim=self.y.size()[1], embed_dim=self.embed_dim)
        y = TopologicalGraphConvolution(adj=self.adj.t(), in_dim=self.x.size()[1], embed_dim=self.embed_dim)
        return nn.ModuleList([x, y])

    def _add_feature_space(self):
        x = FeatureGraphConvolution(sim=self.x_sim, knn=self.x_knn, in_dim=self.x.size()[1],
                                              embed_dim=self.embed_dim)
        y = FeatureGraphConvolution(sim=self.y_sim, knn=self.y_knn, in_dim=self.y.size()[1],
                                              embed_dim=self.embed_dim)
        return nn.ModuleList([x, y])

    def _add_predict_layer(self):
        return CorrelationDecoder(embed_dim=self.embed_dim, kernel_dim=self.kernel_dim, alpha=self.alpha)

    def loss_fun(self, predict: torch.Tensor):
        true_data = torch.masked_select(self.adj, self.mask)
        predict = torch.masked_select(predict, self.mask)
        beta_weight = torch.empty_like(true_data).fill_(self.beta)
        back_betaweight = torch.ones_like(true_data)
        weight = torch.where(true_data.eq(1), beta_weight, back_betaweight)
        bce_loss = nn.BCELoss(weight=weight, reduction="mean")
        return bce_loss(predict, true_data)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 2)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss_pos(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret

    def discriminate(self, z1, z2, sigmoid=True):
        value = torch.matmul(z1, z2.t())
        return torch.sigmoid(value) if sigmoid else value

    def loss_neg(self, pos_z, neg_z):
        pos_loss = -torch.log(
            self.discriminate(pos_z, neg_z, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, pos_z, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def forward(self):
        self_layer_x, self_layer_y = [layer for layer in self.self_space]
        self_x = self_layer_x(x=self.x)
        self_y = self_layer_y(x=self.y)

        feature_layer_x, feature_layer_y = [layer for layer in self.feature_space]
        feature_x = feature_layer_x(x=self.x)
        feature_y = feature_layer_y(x=self.y)

        topo_layer_x, topo_layer_y = [layer for layer in self.topo_space]
        collect_opposite_x, collect_opposite_feature_x = topo_layer_x(opposite_x=self.y, opposite_feature_x=feature_y)
        collect_opposite_y, collect_opposite_feature_y = topo_layer_y(opposite_x=self.x, opposite_feature_x=feature_x)
        feat_x = torch.sum(torch.stack([feature_x], dim=0), dim=0)
        feat_y = torch.sum(torch.stack([feature_y], dim=0), dim=0)
        feat = torch.concat([feat_x, feat_y], dim=0)
        pos_grp = self.gcn_encoder(x=feat, edge_index=self.pos_edge_idx)
        neg_grp = self.gcn_encoder(x=feat, edge_index=self.neg_edge_idx)
        pos_edge_idx_droped = dropout_edge(self.pos_edge_idx, p=0.3)[0]
        pos_feat_droped = drop_feature(feat, 0.1)
        neg_edge_idx_droped = dropout_edge(self.neg_edge_idx, p=0.3)[0]
        neg_feat_droped = drop_feature(feat, 0.1)
        pos_grp1 = self.gcn_encoder(x=pos_feat_droped, edge_index=self.pos_edge_idx)
        pos_grp2 = self.gcn_encoder(x=feat, edge_index=pos_edge_idx_droped)
        neg_grp1 = self.gcn_encoder(x=neg_feat_droped, edge_index=self.neg_edge_idx)
        neg_grp2 = self.gcn_encoder(x=feat, edge_index=neg_edge_idx_droped)
        return pos_grp, pos_grp1, pos_grp2, neg_grp, neg_grp1, neg_grp2


class ModelOptimizer(nn.Module, ABC):
    def __init__(self, model: nn.Module, epochs: int, lr: float, test_data: torch.Tensor,
                 test_mask: torch.Tensor, tolerance=8, frequency=20):
        super(ModelOptimizer, self).__init__()
        self.model = model
        self.epoch = epochs
        self.test_data = test_data
        self.test_mask = test_mask
        self.tolerance = tolerance
        self.frequency = frequency
        self.optim = optim.Adam(model.parameters(), lr=lr)
        self.act = nn.Sigmoid()

    def auc(self, predict: torch.Tensor):
        predict = torch.masked_select(predict, mask=self.test_mask).detach().cpu().numpy()
        true_data = torch.masked_select(self.test_data, mask=self.test_mask).detach().cpu().numpy()
        return roc_auc_score(y_true=true_data, y_score=predict)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        best_predict = 0
        best_auc = 0
        for epoch in range(self.epoch):
            pos_grp, pos_grp1, pos_grp2, neg_grp, neg_grp1, neg_grp2 = self.model()
            self.optim.zero_grad()
            p_grp = pos_grp.to(device='cpu')
            predict = torch.matmul(p_grp[:962, :], p_grp[962:, :].t()).to(device='cuda:0')
            predict = self.act(predict)
            label_loss = self.model.loss_fun(predict=predict)
            pos_loss1 = self.model.loss_pos(pos_grp, pos_grp1)
            pos_loss2 = self.model.loss_pos(pos_grp, pos_grp2)
            neg_loss1 = self.model.loss_neg(pos_grp, neg_grp)
            neg_loss2 = self.model.loss_neg(pos_grp, neg_grp1)
            neg_loss3 = self.model.loss_neg(pos_grp, neg_grp2)

            loss = self.model.eta * label_loss + (1 - self.model.eta) * (pos_loss1 + pos_loss2) / 3 + (
                        1 - self.model.eta) * (neg_loss1 + neg_loss2 + neg_loss3) / 9
            loss.backward()
            self.optim.step()
            auc = self.auc(predict=predict)
            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict, self.test_mask)
            if epoch % self.frequency == 0:
                print("epoch:{:^8d} loss:{:^8.5f} auc:{:^8.5f} best_auc:{:^8.5f}".format(epoch, loss.item(), auc,
                                                                                         best_auc))
        return true_data, best_predict
