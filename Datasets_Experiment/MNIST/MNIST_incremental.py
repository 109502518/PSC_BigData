import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.manifold import SpectralEmbedding
import time as t
import psutil


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


class Dimen_Reduct_Model(nn.Module):
    def __init__(self):
        super(Dimen_Reduct_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(49, 196),
            nn.ReLU(),
            nn.Linear(196, 392),
            nn.ReLU(),
            nn.Linear(392, 196),
            nn.ReLU(),
            nn.Linear(196, 10),
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


# MNIST
train_dataset = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=False)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)


ae = AE()
# device = torch.device("mps")
# ae = ae.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters())
for epoch in range(10):
    running_loss = 0.0
    for x,y in train_loader:
        # x = x.to(device)
        x = x.view(-1, 28 * 28)
        optimizer.zero_grad()
        output = ae(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")


train_num = 60000
test_num = 10000
X_train = train_dataset.data/255
X_train = X_train.view(-1, 28 * 28)
X_train = ae(X_train, stop=True).detach().numpy()
y_train = train_dataset.targets.numpy()
print(X_train.shape)
print(y_train.shape)

X_test = test_dataset.data[:test_num]/255
X_test = X_test.view(-1, 28 * 28)
X_test = ae(X_test, stop=True).detach().numpy()
y_test = test_dataset.targets[:test_num].numpy()

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
print(X.shape)
print(y.shape)


spectralembedding = SpectralEmbedding(n_components=10, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
kmeans = KMeans(n_clusters=10, init='random', n_init='auto')


train_embedding = spectralembedding.fit_transform(X_train)


train_start = t.time()
x = torch.from_numpy(X_train).type(torch.FloatTensor)
u = torch.from_numpy(train_embedding).type(torch.FloatTensor)
dataset = TensorDataset(x, u)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

dimen_reduct_model = Dimen_Reduct_Model()

optimizer = torch.optim.Adam(dimen_reduct_model.parameters())
criterion = nn.MSELoss()
for epoch in range(30):
    running_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = dimen_reduct_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.8f}")


PSC_start = t.time()
PSC_start_memory = psutil.Process().memory_info().rss
x = torch.from_numpy(X).type(torch.FloatTensor)
predict_embedding = dimen_reduct_model(x).detach().numpy()
PSC_index = kmeans.fit_predict(predict_embedding)
PSC_index = PSC_index[train_num:train_num+test_num]
PSC_end = t.time()
PSC_end_memory = psutil.Process().memory_info().rss


PSC_acc = cluster_acc(y_test, PSC_index)
PSC_memory = (PSC_end_memory - PSC_start_memory) / (1024*1024)

psc_ari_score = adjusted_rand_score(y_test, PSC_index)
psc_ami_score = adjusted_mutual_info_score(y_test, PSC_index)


print(f"PSC time(s): {PSC_end - PSC_start:.3f}")
print(f"PSC memory(MB): {PSC_memory:.3f}")
print(f"PSC acc: {PSC_acc:.3f}")
print(f"PSC ARI: {psc_ari_score:.3f}")
print(f"PSC AMI: {psc_ami_score:.3f}")
