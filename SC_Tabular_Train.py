import argparse
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.manifold import SpectralEmbedding


def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--se", default=0, help="Spectral Embedding", type=int)
    args = _parser.parse_args()
    se = args.se
    return se


if __name__ == '__main__':
    se = ParseInput()
    

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
