import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits


class lower_vector(nn.Module):
    def __init__(self):
        super(lower_vector, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x


def SC(X):
    #Similarity Metrix
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


digits = load_digits()
X=digits.data/16
y=digits.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.3)


train_matrix = SC(X_train)
x = torch.from_numpy(X_train).type(torch.FloatTensor)
u = torch.from_numpy(train_matrix).type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(x, u)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

model = lower_vector()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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
print(f"Epoch {epoch+1}, Loss: {loss:.8f}")


X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


x = torch.from_numpy(X).type(torch.FloatTensor)
predict_matirix = model(x).detach().numpy()
kmeans = KMeans(n_clusters=10, init='random', n_init='auto', algorithm='elkan')
cluster3_index = kmeans.fit_predict(predict_matirix)
cluster3_index = cluster3_index[1257:1797]
predict_matirix = predict_matirix[1257:1797]


all_matrix = SC(X)
cluster4_index = kmeans.fit_predict(all_matrix)
cluster4_index = cluster4_index[1257:1797]
all_matrix = all_matrix[1257:1797]


psc_acc = cluster_acc(y_test, cluster3_index)
sc_acc = cluster_acc(y_test, cluster4_index)
sim = cluster_acc(cluster3_index, cluster4_index)

print(f"SC acc: {sc_acc:.3f}")
ari_score = adjusted_rand_score(y_test, cluster4_index)
print(f"SC ARI: {ari_score:.3f}")
ami_score = adjusted_mutual_info_score(y_test, cluster4_index)
print(f"SC AMI: {ami_score:.3f}")

print(f"PSC acc: {psc_acc:.3f}")
ari_score = adjusted_rand_score(y_test, cluster3_index)
print(f"PSC ARI: {ari_score:.3f}")
ami_score = adjusted_mutual_info_score(y_test, cluster3_index)
print(f"PSC AMI: {ami_score:.3f}")

print(f"similarity: {sim:.3f}")
ari_score = adjusted_rand_score(cluster3_index, cluster4_index)
print(f"SC PSC ARI: {ari_score:.3f}")
ami_score = adjusted_mutual_info_score(cluster3_index, cluster4_index)
print(f"SC PSC AMI: {ami_score:.3f}")