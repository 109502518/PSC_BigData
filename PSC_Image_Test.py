import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.manifold import SpectralEmbedding
import argparse


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 1568),
            nn.ReLU(),
            nn.Linear(1568, 784),
            nn.ReLU(),
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 49),
        )
        self.decoder = nn.Sequential(
            nn.Linear(49, 392),
            nn.ReLU(),
            nn.Linear(392, 784),
            nn.ReLU(),
            nn.Linear(784, 1568),
            nn.ReLU(),
            nn.Linear(1568, 784),
        )
        
    def forward(self, x, stop=False):
        x = self.encoder(x)
        if stop:
            return x
        x = self.decoder(x)
        return x


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


def cluster_acc(y, y_pred):
    y = y.astype(np.int64)
    assert y_pred.size == y.size
    D = max(y_pred.max(), y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--ratio", default=1/6, help="Training Data Ratio", type=float)
    _parser.add_argument("--se", default=1, help="Spectral Embedding", type=int)
    _parser.add_argument("--epoch", default=100, help="Epoch", type=int)
    _parser.add_argument("--lr", default=1e-3, help="Learning rate", type=float)
    _parser.add_argument("--batch", default=64, help="Batch size", type=int)
    _parser.add_argument("--layer1", default=196, help="Hidden Layer 1", type=int)
    _parser.add_argument("--layer2", default=392, help="Hidden Layer 2", type=int)
    _parser.add_argument("--layer3", default=196, help="Hidden Layer 3", type=int)
    args = _parser.parse_args()
    ratio, se, epochs, lr, batch_size, layer1, layer2, layer3 = args.ratio, args.se, args.epoch, args.lr, args.batch, args.layer1, args.layer2, args.layer3
    return ratio, se, epochs, lr, batch_size, layer1, layer2, layer3


if __name__ == '__main__':
    ratio, se, epochs, lr, batch_size, layer1, layer2, layer3 = ParseInput()
    
    
    from torchvision.datasets import MNIST
    train_dataset = MNIST(root='data', train=True, transform=ToTensor(), download=False)
    
    
    ae = AE()
    ae.load_state_dict(torch.load('PSC_AE_Model'))
    
    
    all_num = 60000     # MNIST Training Data Num
    X = train_dataset.data/255
    X = X.view(-1, 28 * 28)
    X = ae(X, stop=True).detach().numpy()
    X_train = X[:int(all_num*ratio)]
    y = train_dataset.targets.numpy()
    num_samples, num_features = X.shape
    num_labels = len(set(y))
    print(X_train.shape)


    if se == 0:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='rbf', gamma=1.0)
    elif se == 1:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='nearest_neighbors', n_neighbors=10)

    kmeans = KMeans(n_clusters=10, init='random', n_init='auto')


    PSC_Model = PSC_Mapping_Model(num_features, num_labels, layer1, layer2, layer3)
    PSC_Model.load_state_dict(torch.load('PSC_Image_Model'))

    
    # all_embedding = spectralembedding.fit_transform(X)
    all_embedding = torch.load('Image_all_embedding').numpy()
    SC_index = kmeans.fit_predict(all_embedding)
    
    
    x = torch.from_numpy(X).type(torch.FloatTensor)
    predict_embedding = PSC_Model(x).detach().numpy()
    PSC_index = kmeans.fit_predict(predict_embedding)


    SC_acc = cluster_acc(y, SC_index)
    PSC_acc = cluster_acc(y, PSC_index)
    sim = cluster_acc(SC_index, PSC_index)
    sc_ari_score = adjusted_rand_score(y, SC_index)
    sc_ami_score = adjusted_mutual_info_score(y, SC_index)
    psc_ari_score = adjusted_rand_score(y, PSC_index)
    psc_ami_score = adjusted_mutual_info_score(y, PSC_index)
    sim_ari_score = adjusted_rand_score(SC_index, PSC_index)
    sim_ami_score = adjusted_mutual_info_score(SC_index, PSC_index)


    print(f"SC acc: {SC_acc:.3f}")
    print(f"SC ARI: {sc_ari_score:.3f}")
    print(f"SC AMI: {sc_ami_score:.3f}")
    print(f"PSC acc: {PSC_acc:.3f}")
    print(f"PSC ARI: {psc_ari_score:.3f}")
    print(f"PSC AMI: {psc_ami_score:.3f}")
    print(f"similarity: {sim:.3f}")
    print(f"SC PSC ARI: {sim_ari_score:.3f}")
    print(f"SC PSC AMI: {sim_ami_score:.3f}")