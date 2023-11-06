import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import SpectralEmbedding


class lower_vector(nn.Module):
    def __init__(self):
        super(lower_vector, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
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


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.3)
print(X_train.shape)


# start training
spectralembedding = SpectralEmbedding(n_components=3, affinity='rbf', gamma=1.0, random_state=0)
train_embedding = spectralembedding.fit_transform(X_train)
x = torch.from_numpy(X_train).type(torch.FloatTensor)
u = torch.from_numpy(train_embedding).type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(x, u)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=15, shuffle=True)

model = lower_vector()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()
for epoch in range(100):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        Loss = running_loss/len(dataloader)
print(f"Epoch {epoch+1}, Loss: {Loss:.8f}")
# end training


# combine all data
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


x = torch.from_numpy(X).type(torch.FloatTensor)
predict_embedding = model(x).detach().numpy()
kmeans = KMeans(n_clusters=3, init='random', n_init='auto')
PSC_index = kmeans.fit_predict(predict_embedding)
PSC_index = PSC_index[105:150]  #pick up testing data


all_embedding = spectralembedding.fit_transform(X)
SC_index = kmeans.fit_predict(all_embedding)
SC_index = SC_index[105:150]  #pick up testing data


SC_acc = cluster_acc(y_test, SC_index)
PSC_acc = cluster_acc(y_test, PSC_index)
sim = cluster_acc(SC_index, PSC_index)


print(f"SC acc: {SC_acc:.3f}")
ari_score = adjusted_rand_score(y_test, SC_index)
print(f"SC ARI: {ari_score:.3f}")
ami_score = adjusted_mutual_info_score(y_test, SC_index)
print(f"SC AMI: {ami_score:.3f}")

print(f"PSC acc: {PSC_acc:.3f}")
ari_score = adjusted_rand_score(y_test, PSC_index)
print(f"PSC ARI: {ari_score:.3f}")
ami_score = adjusted_mutual_info_score(y_test, PSC_index)
print(f"PSC AMI: {ami_score:.3f}")

print(f"similarity: {sim:.3f}")
ari_score = adjusted_rand_score(SC_index, PSC_index)
print(f"SC PSC ARI: {ari_score:.3f}")
ami_score = adjusted_mutual_info_score(SC_index, PSC_index)
print(f"SC PSC AMI: {ami_score:.3f}")