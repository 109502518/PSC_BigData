import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding


class lower_vector(nn.Module):
    def __init__(self):
        super(lower_vector, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
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


digits = load_digits()
X=digits.data/16
y=digits.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.3) #1257 + 540 = 1797


spectralembedding = SpectralEmbedding(n_components=10, affinity='nearest_neighbors', n_neighbors=10, random_state=0)
kmeans = KMeans(n_clusters=10, init='random', n_init='auto')


# start training
train_embedding = spectralembedding.fit_transform(X_train)
x = torch.from_numpy(X_train).type(torch.FloatTensor)
u = torch.from_numpy(train_embedding).type(torch.FloatTensor)
dataset = torch.utils.data.TensorDataset(x, u)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

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
print(f"Epoch {epoch+1}, Loss: {Loss:.7f}")
# end training


# combine all data
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


x = torch.from_numpy(X).type(torch.FloatTensor)
predict_embedding = model(x).detach().numpy()
PSC_index = kmeans.fit_predict(predict_embedding)
PSC_index = PSC_index[1257:1797]  #pick up testing data


all_embedding = spectralembedding.fit_transform(X)
SC_index = kmeans.fit_predict(all_embedding)
SC_index = SC_index[1257:1797]  #pick up testing data


PSC_acc = cluster_acc(y_test, PSC_index)
SC_acc = cluster_acc(y_test, SC_index)
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