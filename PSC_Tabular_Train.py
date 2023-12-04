import argparse
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.manifold import SpectralEmbedding


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


def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--ratio", default=0.7, help="Training Data Ratio", type=float)
    _parser.add_argument("--se", default=0, help="Spectral Embedding", type=int)
    _parser.add_argument("--epoch", default=100, help="Epoch", type=int)
    _parser.add_argument("--lr", default=1e-3, help="Learning rate", type=float)
    _parser.add_argument("--batch", default=15, help="Batch size", type=int)
    _parser.add_argument("--layer1", default=32, help="Hidden Layer 1", type=int)
    _parser.add_argument("--layer2", default=64, help="Hidden Layer 2", type=int)
    _parser.add_argument("--layer3", default=32, help="Hidden Layer 3", type=int)
    args = _parser.parse_args()
    ratio, se, epochs, lr, batch_size, layer1, layer2, layer3 = args.ratio, args.se, args.epoch, args.lr, args.batch, args.layer1, args.layer2, args.layer3
    return ratio, se, epochs, lr, batch_size, layer1, layer2, layer3


if __name__ == '__main__':
    ratio, se, epochs, lr, batch_size, layer1, layer2, layer3 = ParseInput()
    

    from sklearn.datasets import load_iris
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    num_samples, num_features = X.shape
    num_labels = len(set(y))
    
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=ratio)
    
    
    if se == 0:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='rbf', gamma=1.0)
    elif se == 1:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='nearest_neighbors', n_neighbors=10)
        

    train_embedding = spectralembedding.fit_transform(X_train)
    x = torch.from_numpy(X_train).type(torch.FloatTensor)
    u = torch.from_numpy(train_embedding).type(torch.FloatTensor)
    dataset = torch.utils.data.TensorDataset(x, u)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
    torch.save(PSC_Model.state_dict(), 'PSC_Tabular_Model')
