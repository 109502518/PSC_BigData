import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from sklearn.manifold import SpectralEmbedding
import argparse


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


def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--se", default=1, help="Spectral Embedding", type=int)
    args = _parser.parse_args()
    se = args.se
    return se


if __name__ == '__main__':
    se = ParseInput()
    
    
    from torchvision.datasets import MNIST
    train_dataset = MNIST(root='data', train=True, transform=ToTensor(), download=False)
    
    
    ae = AE()
    ae.load_state_dict(torch.load('PSC_AE_Model'))
    

    X = train_dataset.data/255
    X = X.view(-1, 28 * 28)
    X = ae(X, stop=True).detach().numpy()
    y = train_dataset.targets.numpy()
    num_labels = len(set(y))


    if se == 0:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='rbf', gamma=1.0)
    elif se == 1:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='nearest_neighbors', n_neighbors=10)

    
    all_embedding = spectralembedding.fit_transform(X)
    torch.save(torch.tensor(all_embedding), 'Image_all_embedding')