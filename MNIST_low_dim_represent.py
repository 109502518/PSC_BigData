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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



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



def avg_row_rmse(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    rmse_list = []
    for row1, row2 in zip(matrix1, matrix2):
        rmse = np.sqrt(np.mean((row1 - row2)**2))
        rmse_list.append(rmse)
    return np.mean(rmse_list)



def avg_row_r2(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    r2_list = []
    for row1, row2 in zip(matrix1, matrix2):
        r2 = r2_score(row1, row2)
        r2_list.append(r2)
    return np.mean(r2_list)



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


train_num = 10000
test_num = 2000
X_train = train_dataset.data[:train_num]/255
X_train = X_train.view(-1, 28 * 28)
X_train = autoencoder(X_train, stop=True).detach().numpy()
y_train = train_dataset.targets[:train_num].numpy()
print(X_train.shape)
print(y_train.shape)

X_test = test_dataset.data[:test_num]/255
X_test = X_test.view(-1, 28 * 28)
X_test = autoencoder(X_test, stop=True).detach().numpy()
y_test = test_dataset.targets[:test_num].numpy()

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
print(X.shape)
print(y.shape)

    
all_matrix = SC_lower_vector(X)
train_matrix = all_matrix[:train_num]

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
        Loss = running_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.4f}")


kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan')


x = torch.from_numpy(X).type(torch.FloatTensor)
predict_matrix = dimen_reduct_model(x).detach().numpy()
all_matrix = all_matrix[train_num:train_num+test_num]
predict_matrix = predict_matrix[train_num:train_num+test_num]


rmse = np.sqrt(mean_squared_error(all_matrix, predict_matrix))
avg_rmse = avg_row_rmse(all_matrix, predict_matrix)
print(f"AVG RMSE = {avg_rmse:.3f}")
print(f"RMSE = {rmse:.3f}")
r2 = r2_score(all_matrix, predict_matrix)
avg_r2 = avg_row_r2(all_matrix, predict_matrix)
print(f"AVG R2 = {avg_r2:.3f}")
print(f"R2 = {r2:.3f}")