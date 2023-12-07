import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.optimize import linear_sum_assignment
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
    train_dataset = MNIST(root='data', train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    
    ae = AE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    for epoch in range(10):
        running_loss = 0.0
        for x,y in train_loader:
            x = x.view(-1, 28 * 28)
            optimizer.zero_grad()
            output = ae(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            Loss = running_loss/len(train_loader)
    torch.save(ae.state_dict(), 'PSC_AE_Model')
    
    
    all_num = 60000     # All of the MNIST Training Data
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


    train_embedding = spectralembedding.fit_transform(X_train)
    x = torch.from_numpy(X_train).type(torch.FloatTensor)
    u = torch.from_numpy(train_embedding).type(torch.FloatTensor)
    dataset = TensorDataset(x, u)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    PSC_Model = PSC_Mapping_Model(num_features, num_labels, layer1, layer2, layer3)
    optimizer = torch.optim.Adam(PSC_Model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = PSC_Model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            Loss = running_loss/len(dataloader)
    torch.save(PSC_Model.state_dict(), 'PSC_Image_Model')
