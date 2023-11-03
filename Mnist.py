import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import time as t
import psutil


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
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
            nn.Linear(32, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x



def SC_lower_vector(X):
    #Similarity Matrix
    dist = cdist(X,X,"euclidean")
    n_neigh=10
    S=np.zeros(dist.shape)
    neigh_index=np.argsort(dist,axis=1)[:,1:n_neigh+1]
    sigma=1
    for i in range(X.shape[0]):
        S[i,neigh_index[i]]=np.exp(-dist[i,neigh_index[i]]/(2*sigma**2))
    S=np.maximum(S,S.T)
    p=10
    #Normalized spectral clustering according to Ng, Jordan, and Weiss
    D=np.diag(np.sum(S,axis=1))
    L=D-S
    D_tmp=np.linalg.inv(D)**(1/2)
    L_sym=np.dot(np.dot(D_tmp,L),D_tmp)
    A,B=np.linalg.eig(L_sym)
    idx=np.argsort(A)[:p]
    U=B[:,idx]
    U=U/((np.sum(U**2,axis=1)**0.5)[:,None])
    return U



def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size



# MNIST 60000 10000 1x28x28
train_dataset = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=False)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


autoencoder = Autoencoder()
# device = torch.device("mps")
# autoencoder = autoencoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
for epoch in range(100):
    running_loss = 0.0
    for x,y in train_loader:
        # x = x.to(device)
        x = x.view(-1, 28 * 28)
        optimizer.zero_grad()
        output = autoencoder(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")


all_num = 30000
train_num = 10000
X = train_dataset.data[:all_num]/255
X = X.view(-1, 28 * 28)
X = autoencoder(X, stop=True).detach().numpy()
X_train = X[:train_num]
y_true = train_dataset.targets[:all_num].numpy()
print(X_train.shape)


train_start = t.time()
train_matrix = SC_lower_vector(X_train)
x = torch.from_numpy(X_train).type(torch.FloatTensor)
u = torch.from_numpy(train_matrix).type(torch.FloatTensor)
dataset = TensorDataset(x, u)
dataloader = DataLoader(dataset, batch_size=40, shuffle=True)

dimen_reduct_model = Dimen_Reduct_Model()
optimizer = torch.optim.Adam(dimen_reduct_model.parameters(), lr=0.001)
criterion = nn.MSELoss()
for epoch in range(200):
    running_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = dimen_reduct_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")

train_end = t.time()
print(f"training time: {train_end - train_start:.8f}")


kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan')


SC_start = t.time()
all_matrix = SC_lower_vector(X)
cluster3_index = kmeans.fit_predict(all_matrix)
cluster3_index = cluster3_index[train_num:all_num]
SC_end = t.time()
print(f"30000 data SC time: {SC_end - SC_start:.8f}")


PSC_start = t.time()
PSC_start_memory = psutil.Process().memory_info().rss
x = torch.from_numpy(X).type(torch.FloatTensor)
predict_matrix = dimen_reduct_model(x).detach().numpy()
cluster4_index = kmeans.fit_predict(predict_matrix)
cluster4_index = cluster4_index[train_num:all_num]
PSC_end = t.time()
PSC_end_memory = psutil.Process().memory_info().rss


SC_acc = cluster_acc(y_true, cluster3_index)
PSC_acc = cluster_acc(y_true, cluster4_index)
sim = cluster_acc(cluster3_index, cluster4_index)
PSC_memory = (PSC_end_memory - PSC_start_memory) / (1024*1024)


sc_ari_score = adjusted_rand_score(y_true, cluster3_index)
sc_ami_score = adjusted_mutual_info_score(y_true, cluster3_index)
psc_ari_score = adjusted_rand_score(y_true, cluster4_index)
psc_ami_score = adjusted_mutual_info_score(y_true, cluster4_index)
sim_ari_score = adjusted_rand_score(cluster3_index, cluster4_index)
sim_ami_score = adjusted_mutual_info_score(cluster3_index, cluster4_index)


print(f"SC acc: {SC_acc:.3f}")
print(f"SC ARI: {sc_ari_score:.3f}")
print(f"SC AMI: {sc_ami_score:.3f}")

print(f"PSC time(s): {PSC_end - PSC_start:.3f}")
print(f"PSC memory(MB): {PSC_memory:.3f}")
print(f"PSC acc: {PSC_acc:.3f}")
print(f"PSC ARI: {psc_ari_score:.3f}")
print(f"PSC AMI: {psc_ami_score:.3f}")

print(f"similarity: {sim:.3f}")
print(f"SC PSC ARI: {sim_ari_score:.3f}")
print(f"SC PSC AMI: {sim_ami_score:.3f}")