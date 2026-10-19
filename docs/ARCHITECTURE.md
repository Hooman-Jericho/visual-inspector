# Visual Inspector — Architecture & System Design

Visual Inspector is an industrial-grade, unsupervised defect detection engine implementing **PatchCore** (Roth et al., CVPR 2022) faithfully from scratch in PyTorch.

---

## 1. High-Level Pipeline

The system is organized into three decoupled pipelines: **Data Pipeline**, **Inference / Localization Engine**, and **Serving Layer**.

```mermaid
flowchart TD
    subgraph Data["1. Data Pipeline"]
        raw[Raw Industrial Images] --> prep[Resize 256 -> Center Crop 224]
        prep --> norm[ImageNet Standardization]
        norm --> split[90% Training-Good / 10% Val-Good]
    end

    subgraph Feature["2. PatchCore Feature Extraction"]
        split --> backbone["Pretrained WideResNet-50-2<br/>(Frozen, Eval Mode)"]
        backbone --> |hook| l2[Layer 2 Features]
        backbone --> |hook| l3[Layer 3 Features]
        l2 --> p2[AvgPool2d 3x3 s=1 p=1]
        l3 --> p3[AvgPool2d 3x3 s=1 p=1]
        p3 --> up[Bilinear Upsample to Layer 2]
        p2 & up --> cat[Channel Concatenation]
        cat --> pool[Adaptive Channel Pool to d=1024]
        pool --> patches["Patch Memory Embeddings<br/>(N x 1024)"]
    end

    subgraph Memory["3. Coreset & Memory Bank"]
        patches --> proj[Random Gaussian Projection to 128-d]
        proj --> kcenter["Greedy k-Center Subsampling<br/>(Ratio r = 10%)"]
        kcenter --> bank["Optimized Memory Bank<br/>(0.10 x N x 1024)"]
        bank --> index["Faiss L2 Index / Torch cdist Fallback"]
    end

    subgraph Inference["4. Anomaly Scoring & Heatmap"]
        query[Query Image] --> q_patches[Extract Query Patches]
        q_patches --> index
        index --> nn_dist["Patch Nearest Neighbor Distances"]
        nn_dist --> grid["Reshape to Spatial Grid (H x W)"]
        grid --> bi_up["Bilinear Interpolate to (224 x 224)"]
        bi_up --> blur["Gaussian Filter (sigma=4.0)"]
        blur --> amap["Pixel-level Anomaly Map"]
        amap --> iscore["Image Anomaly Score = max(Map)"]
    end

    subgraph Serving["5. Deployment & API Layer"]
        iscore --> api["FastAPI Production API"]
        amap --> api
        api --> ui["Single-File Demo Web UI"]
        api --> client["External REST Client / Factory PLC"]
    end
```

---

## 2. Core Components

### 2.1 Feature Extractor (`models/backbone.py`)
- **Backbone**: Wide ResNet-50-2 (`torchvision.models.wide_resnet50_2`), frozen in `eval()` mode.
- **Hook mechanism**: Uses PyTorch `RemovableHandle` forward hooks registered on `layer2` and `layer3`.
- **Zero gradient overhead**: Sets `param.requires_grad_(False)` on all parameters and runs forward passes inside `torch.no_grad()`.

### 2.2 Local Neighborhood Aggregation (`models/patchcore.py`)
- Aggregates spatial neighborhoods using `nn.AvgPool2d(3, stride=1, padding=1)` on each feature map.
- Bilinearly upsamples `layer3` to match `layer2` spatial resolution.
- Concatenates channels and adaptively pools across channel dimension to target dimension $d=1024$.
- Flattens into patch embeddings $(B \times H \times W, 1024)$.

### 2.3 Greedy k-Center Coreset Selection (`models/coreset.py`)
- Subsamples nominal training patch embeddings by ratio $r \in [0.01, 0.10, 1.00]$.
- **Random Projection**: Projects embeddings from 1024-d down to 128-d via fixed Gaussian projection matrix for distance computations, drastically accelerating coreset selection.
- **Batched Distance Updates**: Computes pairwise distances in chunks of 4,096 points to bound CPU/GPU memory consumption.

### 2.4 Unified kNN Engine (`models/knn.py`)
- **`FaissKNN`**: Fast exact nearest neighbor search via `faiss.IndexFlatL2`.
- **`TorchKNN`**: Pure PyTorch batched `torch.cdist` fallback requiring zero C++ compilation or native libraries.
- Standardized Euclidean distance output across both backends.

### 2.5 Serving & Model Registry (`serve/`)
- **Lazy loading**: Models are loaded on-demand from `artifacts/{category}/` on the first incoming request and cached in memory.
- **Safe deserialization**: Only tensor weights are loaded via `torch.load(..., weights_only=True)` and metadata is stored in plaintext JSON (`meta.json`).
