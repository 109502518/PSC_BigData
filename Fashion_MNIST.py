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


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 7, 1, 3),
        )


    def forward(self, x, stop=False):
        x = self.encoder(x)
        if stop:
            x = x.view(x.size(0), -1)
            return x
        x = self.decoder(x)
        return x



class Dimen_Reduct_Model(nn.Module):
    def __init__(self):
        super(Dimen_Reduct_Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(49, 196),
            nn.ReLU(inplace=False),
            nn.Linear(196, 392),
            nn.ReLU(inplace=False),
            nn.Linear(392, 196),
            nn.ReLU(inplace=False),
            nn.Linear(196, 10)
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
    k=10

    #Normalized spectral clustering according to Ng, Jordan, and Weiss
    D=np.diag(np.sum(S,axis=1))
    L=D-S
    D_tmp=np.linalg.inv(D)**(1/2)
    L_sym=np.dot(np.dot(D_tmp,L),D_tmp)
    A,B=np.linalg.eig(L_sym)
    idx=np.argsort(A)[:k]
    train_matrix=B[:,idx]
    train_matrix=train_matrix/((np.sum(train_matrix**2,axis=1)**0.5)[:,None])
    return train_matrix



def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size



train_dataset = datasets.FashionMNIST(root='data', train=True, transform=ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.FashionMNIST(root='data', train=False, transform=ToTensor(), download=False)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


autoencoder = Autoencoder()

# device = torch.device("mps")
# autoencoder = autoencoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
for epoch in range(30):
    running_loss = 0.0
    for x,y in train_loader:
        # x = x.to(device)
        optimizer.zero_grad()
        output = autoencoder(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")


all_num = 30000
X = train_dataset.data[:all_num]/255
X = X.unsqueeze(1) 
X = X.to(torch.float32)
X = autoencoder(X, stop=True).detach().numpy()
y = train_dataset.targets[:all_num].numpy()
print(X.shape)
print(y.shape)
all_matrix = SC_lower_vector(X)

X_train = X[:10000]
train_matrix = all_matrix[:10000]
train_matrix = SC_lower_vector(x)


dimen_reduct_model = Dimen_Reduct_Model()
x = torch.from_numpy(x).type(torch.FloatTensor)
u = torch.from_numpy(train_matrix).type(torch.FloatTensor)
dataset = TensorDataset(x, u)
dataloader = DataLoader(dataset, batch_size=40, shuffle=True)
optimizer = torch.optim.Adam(dimen_reduct_model.parameters(), lr=0.001)
criterion = nn.MSELoss()
for epoch in range(100):
    running_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = dimen_reduct_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")


kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan')


cluster1_index = kmeans.fit_predict(all_matrix)
X = torch.from_numpy(X).type(torch.FloatTensor)
predict_matrix = dimen_reduct_model(X).detach().numpy()
cluster2_index = kmeans.fit_predict(predict_matrix)

SC_acc = cluster_acc(y, cluster1_index)
PSC_acc = cluster_acc(y, cluster2_index)
sim = cluster_acc(cluster1_index, cluster2_index)
sc_ari_score = adjusted_rand_score(y, cluster1_index)
sc_ami_score = adjusted_mutual_info_score(y, cluster1_index)
psc_ari_score = adjusted_rand_score(y, cluster2_index)
psc_ami_score = adjusted_mutual_info_score(y, cluster2_index)
sim_ari_score = adjusted_rand_score(cluster1_index, cluster2_index)
sim_ami_score = adjusted_mutual_info_score(cluster1_index, cluster2_index)

print(f"SC acc: {SC_acc:.3f}")
print(f"SC ARI: {sc_ari_score:.3f}")
print(f"SC AMI: {sc_ami_score:.3f}")

print(f"PSC acc: {PSC_acc:.3f}")
print(f"PSC ARI: {psc_ari_score:.3f}")
print(f"PSC AMI: {psc_ami_score:.3f}")

print(f"similarity: {sim:.3f}")
print(f"SC PSC ARI: {sim_ari_score:.3f}")
print(f"SC PSC AMI: {sim_ami_score:.3f}")