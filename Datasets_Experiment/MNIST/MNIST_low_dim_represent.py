import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


def avg_row_rmse(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    num_rows, num_cols = matrix1.shape
    rmse_list = []
    for i in range(num_rows):
        rmse = np.sqrt(mean_squared_error(matrix1[i], matrix2[i]))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


def avg_row_r2(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    r2_list = []
    for row1, row2 in zip(matrix1, matrix2):
        r2 = r2_score(row1, row2)
        r2_list.append(r2)
    return np.mean(r2_list)



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


train_num = 10000
test_num = 2000
X_train = train_dataset.data[:train_num]/255
X_train = X_train.view(-1, 28 * 28)
X_train = ae(X_train, stop=True).detach().numpy()
y_train = train_dataset.targets[:train_num].numpy()
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


all_embedding = spectralembedding.fit_transform(X)
train_embedding = all_embedding[:train_num]


x = torch.from_numpy(X_train).type(torch.FloatTensor)
u = torch.from_numpy(train_embedding).type(torch.FloatTensor)
dataset = TensorDataset(x, u)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

dimen_reduct_model = Dimen_Reduct_Model()

optimizer = torch.optim.Adam(dimen_reduct_model.parameters())
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
        Loss = running_loss/len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {Loss:.8f}")


x = torch.from_numpy(X).type(torch.FloatTensor)
predict_embedding = dimen_reduct_model(x).detach().numpy()
predict_embedding = predict_embedding[train_num:train_num+test_num]
test_embedding = all_embedding[train_num:train_num+test_num]


avg_rmse = avg_row_rmse(test_embedding, predict_embedding)
print(f"AVG RMSE = {avg_rmse:.5f}")
avg_r2 = avg_row_r2(test_embedding, predict_embedding)
print(f"AVG R2 = {avg_r2:.3f}")
