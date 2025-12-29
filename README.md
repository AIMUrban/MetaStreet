# MetaStreet

This is the PyTorch implementation of MetaStreet: Semi-Supervised Multimodal Learning for Street-Level Socioeconomic Prediction.

## Project Structure

### Core Files

- **`main.py`** - Entry point for training the MetaStreet model. Handles argument parsing, device configuration, and evaluation for downstream tasks.

- **`model.py`** - Contains the main MetaStreet model implementation:
  - `SV_GAT` class: Main model with multimodal representation learning and semi-supervised graph contrastive learning

- **`AdjacencyMatrix.py`** - Computes neighborhood relationships from segmented street view images and generates spatial adjacency matrices.

- **`word2vec.py`** - Trains semantic proportion vectors for object categories.

- **`ImageRepresentation.py`** - Combines spatial adjacency matrices, semantic proportion vectors, and category embeddings to compute and save image representations.

### Dataset Structure

```
data/
├── wuhan/                          # Wuhan dataset
│   ├── length.npy                  # Street segment lengths
│   ├── edge_index.pt               # Graph edge connections
│   ├── function/                   # Street function classification task
│   │   ├── label_all_function.npy
│   │   ├── label_mask.npy
│   │   └── test_mask.pt
│   └── poi/                        # POI classification task
│       ├── label_all_poi_level.npy
│       ├── label_mask_poi_level.npy
│       └── test_mask_poi_level.npy
│
├── xian/                           # Xi'an dataset
│   ├── length.npy                  # Street segment lengths
│   ├── edge_index.pt               # Graph edge connections
│   ├── house/                      # House price prediction task
│   │   ├── label_all_house_level.npy
│   │   ├── label_mask_house_level.npy
│   │   └── test_mask_house_level.npy
│   └── poi/                        # POI classification task
│       ├── label_all_poi_level.npy
│       ├── label_mask_poi_level.npy
│       └── test_mask_poi_level.npy
```

## Supported Tasks by City

| City   | Available Tasks |
|:------:|:---------------:|
| Wuhan  | function, poi   |
| Xi'an  | house, poi      |

## Usage

### Basic Usage

```bash
# Train on Wuhan dataset (runs all available tasks: function, poi)
python main.py --city wuhan --device cuda:0

# Train on Xi'an dataset (runs all available tasks: house, poi)
python main.py --city xian --device cuda:0
```

### Specify Downstream Task

```bash
# Train for street function prediction (Wuhan only)
python main.py --city wuhan --downstream function --device cuda:0

# Train for POI classification (both cities)
python main.py --city wuhan --downstream poi --device cuda:0
python main.py --city xian --downstream poi --device cuda:0

# Train for house price prediction (Xi'an only)
python main.py --city xian --downstream house --device cuda:0
```