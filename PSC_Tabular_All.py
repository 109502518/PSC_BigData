import argparse
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.manifold import SpectralEmbedding


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
    

    if se == 0:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='rbf', gamma=1.0)
    elif se == 1:
        spectralembedding = SpectralEmbedding(n_components=num_labels, affinity='nearest_neighbors', n_neighbors=10)



    all_embedding = spectralembedding.fit_transform(X)
    torch.save(torch.tensor(all_embedding), 'Tabular_all_embedding')
