# coding: utf-8
import os
import sys
import time

sys.setrecursionlimit(10000)  # 例如这里设置为一万
sys.path.append(os.path.dirname(__file__))
from GDSC.api.sampler import MultipleTargetSampler
from GDSC.api.model import BiSpaceGraphConvolution, ModelOptimizer
from GDSC.api.utils import *

dtype = torch.float32
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_path = "./processed_data/"
target_drug_cids = np.array([5330286, 11338033, 24825971], dtype=np.int32)

# 加载细胞系-药物矩阵
print("Reading cell drug binary ......")
cell_drug = pd.read_csv(data_path + "cell_drug_binary.csv", index_col=0, header=0)
drug_cids = cell_drug.columns.values.astype(np.int32)
cell_drug = np.array(cell_drug, dtype=np.float32)
cell_drug = np.nan_to_num(cell_drug)

target_indexes = common_data_index(drug_cids, target_drug_cids)

# 加载药物-指纹特征矩阵
print("Reading drug feature.....")
drug_feature = pd.read_csv(data_path + "drug_feature.csv", index_col=0, header=0)
drug_feature = torch.from_numpy(np.array(drug_feature)).to(dtype=dtype, device=device)
drug_sim = jaccard_coef(tensor=drug_feature)

# 加载细胞系-基因特征矩阵
print("Reading gene ......")
gene = pd.read_csv(data_path + "cell_gene.csv", index_col=0, header=0)
gene = torch.from_numpy(np.array(gene)).to(dtype=dtype, device=device)
cell_sim = calculate_gene_exponent_similarity(x=gene, mu=3)

# 加载null_mask
print("Reading null mask ......")
null_mask = pd.read_csv(data_path + "null_mask.csv", index_col=0, header=0)
null_mask = np.array(null_mask, dtype=np.float32)

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()
# eta_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
times = []
aucs = [];
f1s = [];
accs = [];
aps = [];
mccs = []
eta_auc = []
for fold in range(3):
    start = time.time()
    pos_edge = []
    neg_edge = []
    cal = False
    sampler = MultipleTargetSampler(adj=cell_drug, null_mask=null_mask, indexes=target_indexes)
    eta_auc_item = []
    for train_data, test_data, train_mask, test_mask in sampler(dtype=dtype, device=device):
        if not cal:
            for i, row in enumerate(train_data):
                for j, v in enumerate(row):
                    if v == 1:
                        pos_edge.append([i, 962 + j])
                    else:
                        neg_edge.append([i, 962 + j])
            cal = True
            pos_edge = list(map(list, zip(*pos_edge)))
            neg_edge = list(map(list, zip(*neg_edge)))
        print("FLAG!\n")
        model = BiSpaceGraphConvolution(adj=train_data, x_sim=cell_sim, y_sim=drug_sim, mask=train_mask,
                                        embed_dim=224, kernel_dim=192, alpha=6.9,
                                        x_knn=7, y_knn=7,
                                        eta=0.7,
                                        pos_edge_idx=torch.tensor(pos_edge, device='cuda:0'),
                                        neg_edge_idx=torch.tensor(neg_edge, device='cuda:0')).to(device)
        opt = ModelOptimizer(model=model, epochs=300, lr=5e-5, test_data=test_data, test_mask=test_mask)
        true_data, predict_data = opt()
        true_datas = true_datas._append(to_data_frame(data=true_data))
        predict_datas = predict_datas._append(to_data_frame(data=predict_data))
        aucs.append(roc_auc(true_data=true_data, predict_data=predict_data))
        f1s.append(f1_res(true_data=true_data, predict_data=predict_data))
        accs.append(acc_score(true_data=true_data, predict_data=predict_data))
        aps.append(ap_score(true_data=true_data, predict_data=predict_data))
        mccs.append(mcc_score(true_data=true_data, predict_data=predict_data))
        eta_auc_item.append(roc_auc(true_data=true_data, predict_data=predict_data))
    end = time.time()
    times.append(end - start)
    eta_auc.append(eta_auc_item)
print("Times:", np.mean(times))
print("AUCS:", np.mean(aucs))
print("F1s:", np.max(f1s))
print("ACCS:", np.max(accs))
print("APS:", np.max(aps))
print("MCCS:", np.max(mccs))
print(np.mean(eta_auc, axis=1))

from GDSC.api.plot import ro_curve

ro_curve(true_datas.values.flatten(), predict_datas.values.flatten())
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")
