import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


class PSC_Mapping_Model(nn.Module):
    def __init__(self, feature, vector, hid_layer1, hid_layer2, hid_layer3):
        super(PSC_Mapping_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature, hid_layer1),
            nn.ReLU(),
            nn.Linear(hid_layer1, hid_layer2),
            nn.ReLU(),
            nn.Linear(hid_layer2, hid_layer3),
            nn.ReLU(),
            nn.Linear(hid_layer3, vector)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--layer1", default=32, help="Hidden Layer 1", type=int)
    _parser.add_argument("--layer2", default=64, help="Hidden Layer 2", type=int)
    _parser.add_argument("--layer3", default=32, help="Hidden Layer 3", type=int)
    args = _parser.parse_args()
    layer1, layer2, layer3 = args.layer1, args.layer2, args.layer3
    return layer1, layer2, layer3


if __name__ == '__main__':
    layer1, layer2, layer3 = ParseInput()
    

    from sklearn.datasets import load_iris
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    num_samples, num_features = X.shape
    num_labels = len(set(y))
    
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    
    kmeans = KMeans(n_clusters=num_labels, init='random', n_init='auto')


    PSC_Model = PSC_Mapping_Model(num_features, num_labels, layer1, layer2, layer3)
    PSC_Model.load_state_dict(torch.load('PSC_Tabular_Model'))


    x = torch.from_numpy(X).type(torch.FloatTensor)
    predict_embedding = PSC_Model(x).detach().numpy()
    PSC_index = kmeans.fit_predict(predict_embedding)


    all_embedding = torch.load('Tabular_all_embedding').numpy()
    SC_index = kmeans.fit_predict(all_embedding)


    SC_acc = cluster_acc(y, SC_index)
    PSC_acc = cluster_acc(y, PSC_index)
    sim = cluster_acc(SC_index, PSC_index)


    print(f"SC acc: {SC_acc:.3f}")
    ari_score = adjusted_rand_score(y, SC_index)
    print(f"SC ARI: {ari_score:.3f}")
    ami_score = adjusted_mutual_info_score(y, SC_index)
    print(f"SC AMI: {ami_score:.3f}")

    print(f"PSC acc: {PSC_acc:.3f}")
    ari_score = adjusted_rand_score(y, PSC_index)
    print(f"PSC ARI: {ari_score:.3f}")
    ami_score = adjusted_mutual_info_score(y, PSC_index)
    print(f"PSC AMI: {ami_score:.3f}")

    print(f"similarity: {sim:.3f}")
    ari_score = adjusted_rand_score(SC_index, PSC_index)
    print(f"SC PSC ARI: {ari_score:.3f}")
    ami_score = adjusted_mutual_info_score(SC_index, PSC_index)
    print(f"SC PSC AMI: {ami_score:.3f}")
