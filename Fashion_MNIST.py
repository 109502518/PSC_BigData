import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 5, 2, 2, 1),
            nn.Sigmoid(),
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


#Fashion-MNIST
train_dataset = datasets.FashionMNIST(root='data', train=True, transform=ToTensor(), download=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.FashionMNIST(root='data', train=False, transform=ToTensor(), download=False)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

autoencoder = Autoencoder()

# device = torch.device("mps")
# autoencoder = autoencoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters())
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


all_num = 60000
train_num = 20000

X = train_dataset.data[:all_num]/255
X = X.unsqueeze(1)
X = X.to(torch.float32)
X = autoencoder(X, stop=True).detach().numpy()
y_true = train_dataset.targets[:all_num].numpy()
print(X.shape)
print(y_true.shape)

X_train = X[:train_num]
y_train = y_true[:train_num]
print(X_train.shape)
print(y_train.shape)


spectralembedding = SpectralEmbedding(n_components=10, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
kmeans = KMeans(n_clusters=10, init='random', n_init='auto')


all_embedding = spectralembedding.fit_transform(X)

train_embedding = spectralembedding.fit_transform(X_train)

x = torch.from_numpy(X_train).type(torch.FloatTensor)
u = torch.from_numpy(train_embedding).type(torch.FloatTensor)
dataset = TensorDataset(x, u)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

dimen_reduct_model = Dimen_Reduct_Model()

optimizer = torch.optim.Adam(dimen_reduct_model.parameters())
criterion = nn.MSELoss()
for epoch in range(45):
    running_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = dimen_reduct_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.7f}")


sc_index = kmeans.fit_predict(all_embedding)


x = torch.from_numpy(X).type(torch.FloatTensor)
predict_embedding = dimen_reduct_model(x).detach().numpy()
psc_index = kmeans.fit_predict(predict_embedding)


SC_acc = cluster_acc(y_true, sc_index)
PSC_acc = cluster_acc(y_true, psc_index)
sim = cluster_acc(sc_index, psc_index)
sc_ari_score = adjusted_rand_score(y_true, sc_index)
sc_ami_score = adjusted_mutual_info_score(y_true, sc_index)
psc_ari_score = adjusted_rand_score(y_true, psc_index)
psc_ami_score = adjusted_mutual_info_score(y_true, psc_index)
sim_ari_score = adjusted_rand_score(sc_index, psc_index)
sim_ami_score = adjusted_mutual_info_score(sc_index, psc_index)


print(f"SC acc: {SC_acc:.3f}")
print(f"SC ARI: {sc_ari_score:.3f}")
print(f"SC AMI: {sc_ami_score:.3f}")
print(f"PSC acc: {PSC_acc:.3f}")
print(f"PSC ARI: {psc_ari_score:.3f}")
print(f"PSC AMI: {psc_ami_score:.3f}")
print(f"similarity: {sim:.3f}")
print(f"SC PSC ARI: {sim_ari_score:.3f}")
print(f"SC PSC AMI: {sim_ami_score:.3f}")
