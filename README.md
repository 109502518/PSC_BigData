# Parametric Spectral Clustering

## Introduction
This repository hosts the implementation for the paper: Toward Efficient and Incremental Spectral Clustering via Parametric Spectral Clustering, published at IEEE International Conference on Big Data 2023.

The project addresses the significant computational challenges in traditional Spectral Clustering, particularly when integrating new data. Traditional methods require reprocessing the entire dataset, which is impractical for large-scale applications. Our innovative approach, Parametric Spectral Clustering (PSC), optimizes the essential steps of Spectral Clustering. This method significantly enhances efficiency, allowing for large dataset processing and facilitating incremental clustering without the need to reprocess entire datasets.

## Experiment Environment

- Python 3.10.8 in MacOS 14.1.1
- Install the required packages by
```
pip install -r requirements.txt
```

## Experiment Datasets
- Tabular Datasets
    - Iris
    - Wine
    - BreastCancer
- Image Datasets
    - UCIHW
    - MNIST
    - Fashion-MNIST

## Quickstart

### Tabular Dataset, take Iris dataset for example

#### PSC Model Training
- Usage :
```
python3 PSC_Tabular_Train.py
        --ratio < 0 ~ 1 (default:0.7)>
        --se <Choose the Spectral Embedding Affinity Matrix, 0 for rbf, 1 for nearest_neighbors (default:0)>
        --epoch <epoch (default:100)>
        --lr <Learning Rate (default:0.001)>
        --batch <Batch Size (default:15)>
        --layer1 <PSC Model Hidden layer1 (default:32)>
        --layer2 <PSC Model Hidden layer2 (default:64)>
        --layer3 <PSC Model Hidden layer3 (default:32)>
```
- Example :
```
python3 PSC_Tabular_Train.py \
        --ratio 0.7 \
        --se 0 \
        --epoch 100 \
        --lr 0.001 \
        --batch 15 \
        --layer1 32 \
        --layer2 64 \
        --layer3 32
``` 

#### To compare the similarity, compute the Spectral Clustering with all data
```
python3 PSC_Tabular_All.py --se 0
```

#### PSC Model Testing
```
python3 PSC_Tabular_Test.py \
        --layer1 32 \
        --layer2 64 \
        --layer3 32
```

---

### Image Dataset, take MNIST dataset for example

#### PSC Model Training

- Usage :
```
$ python3 PSC_Image_Train.py
        --ratio < 0 ~ 1 (default:1/6)>
        --se <Choose the Spectral Embedding Affinity Matrix, 0 for rbf, 1 for nearest_neighbors (default:1)>
        --epoch <epoch (default:100)>
        --lr <Learning Rate (default:0.001)>
        --batch <Batch Size (default:64)>
        --layer1 <PSC Model Hidden layer1 (default:196)>
        --layer2 <PSC Model Hidden layer2 (default:392)>
        --layer3 <PSC Model Hidden layer3 (default:196)>
```
- Example :
```
python3 PSC_Image_Train.py \
        --ratio 1/6 \
        --se 1 \
        --epoch 100 \
        --lr 0.001 \
        --batch 64 \
        --layer1 196 \
        --layer2 392 \
        --layer3 196
``` 

#### To compare the similarity, compute the Spectral Clustering with all data
```
python3 PSC_Image_All.py --se 1
```

#### PSC Model Testing
```
python3 PSC_Image_Test.py \
        --layer1 196 \
        --layer2 392 \
        --layer3 196
```

## Citation
Please cite the paper if you find the work useful.

    @inproceedings{chen23toward,
        title={Toward Efficient and Incremental Spectral Clustering via Parametric Spectral Clustering},
        author={Jo-Chun Chen and Hung-Hsuan Chen},
        booktitle={IEEE International Conference on Big Data},
        year={2023}
    }
