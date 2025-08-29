# 🏥 PFedCDP: Privacy-Preserving Heterogeneity-Aware Personalized Federated Learning for Medical Imaging

PFedCDP is a novel **Personalized Federated Learning (PFL) framework** designed to address the combined challenges of **device heterogeneity, statistical heterogeneity (non-IID data), privacy preservation, and communication efficiency**.  
Our framework is specifically evaluated on **medical imaging datasets** to demonstrate its effectiveness in sensitive healthcare applications.

---

## ✨ Key Features

- **Heterogeneity-Aware Client Clustering**:  
  Clients are grouped into High, Medium, and Low capability clusters based on CPU, memory, battery, and network resources. Each cluster is assigned a model architecture suited to its capability.

- **Fisher Information-Guided Personalization**:  
  Preserves locally important parameters while replacing less critical ones with global knowledge, balancing personalization and generalization.

- **Knowledge Distillation (KD)**:  
  - *Client-Side*: The server’s classifier acts as a **teacher** guiding each client’s classifier.  
  - *Server-Side*: Cluster-aggregated models serve as an **ensemble of teachers** for post-hoc refinement.

- **Shapley Value-Weighted Aggregation**:  
  Provides interpretable and fair weighting of teacher contributions in server-side KD.

- **Supervised Contrastive Learning (SCL)**:  
  Improves representation learning by encouraging intra-class compactness and inter-class separation during refinement.

- **Privacy + Communication Efficiency**:  
  Integrates **Differential Privacy (DP)** with Gaussian noise and **Quantization** to ensure secure, lightweight communication.

---

## ⚙️ Framework Architecture

The PFedCDP framework operates in **two main phases**:  
1. **Iterative Client-Side Training Loop** (personalization + selective parameter sharing).  
2. **Server-Side Post-Hoc Refinement** (MTKD + Shapley weighting + SCL).

<p align="center">
  <img src="https://github.com/shadhin39/PFedCDP/blob/main/Group%20167.png?raw=true" width="90%">
</p>  

**Figure (a)**: High-level PFedCDP architecture with heterogeneous client clusters and server refinement.  
<p align="center">
  <img src="https://github.com/shadhin39/PFedCDP/blob/main/Group%20168.png?raw=true" width="90%">
</p>  
**Figure (b)**: Iterative training loop with Fisher-based personalization and KD.

---

## 📦 Datasets

We simulate realistic medical imaging with **two mammography datasets**:

- **Private Client Data (non-IID)**:  
  [CBIS-DDSM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629)  
  Partitioned across clients using **Dirichlet distribution ($\alpha=0.5$)** to mimic statistical heterogeneity.

- **Public Server Data**:  
  [MIAS](https://www.repository.cam.ac.uk/items/1c7f65f5-77b0-49db-9e7a-d65fb3afeb6d)  
  Used only for post-hoc refinement to improve generalization without exposing private data.

---

## 🔧 Dependencies

The implementation is built with the **[Flower](https://flower.dev/)** federated learning framework.

```
Python >= 3.10  
TensorFlow >= 2.15 or PyTorch >= 2.0  
flwr  
numpy  
pandas  
matplotlib  
shap
```

---

## ⚙️ Parameters

| Argument             | Description                                | Value |
|----------------------|--------------------------------------------|-------|
| `federated_rounds`   | Number of federated training rounds        | 20    |
| `local_epochs`       | Local training epochs per round            | 20    |
| `alpha`              | Dirichlet parameter (non-IID degree)       | 0.5   |
| `epsilon`            | Privacy budget (DP)                        | 5.0   |
| `C2`                 | $L_2$ norm clipping threshold (DP)         | 2.5   |
| `quantization_bits`  | Quantization bits for updates              | 8     |
| `T_KD`               | Temperature for knowledge distillation     | Varies|
| `lambda`             | Balance factor for $L_{CE}$ and $L_{KD}$   | Varies|

---

## 🚀 Usage

### 1. Run PFedCDP
- **Server**
```bash
python server_main_contrastive_learning.py
```

- **Clients (Clustered by capability)**
```bash
# High Capability Client
python client.py --port=8080  

# Medium Capability Client
python client1.py --port=8081  

# Low Capability Client
python client2.py --port=8082  
```

### 2. Run FedAvg (Baseline)
- **Server**
```bash
python server.py
```

- **Clients**
```bash
python client.py
python client.py
```

---

## 📊 Results

PFedCDP achieves strong personalization and privacy preservation under heterogeneity, outperforming FedAvg and approaching centralized performance.

| Method                   | CBIS-DDSM (Private Test %) | MIAS (Public Test %) |
|---------------------------|:-------------------------:|:-------------------:|
| Centralized (Upper Bound) | 96.2                      | 99.8                |
| FedAvg (Baseline)         | 85.7                      | 91.3                |
| **PFedCDP (Ours)**        | **94.1**                  | **99.5**            |

---

## 📝 Citation
If you use this work, please cite:

```bibtex
@article{shadin2025pfedcdp,
  title={Heterogeneity-Aware Private Personalized Federated Learning for Medical Imaging via Contrastive Distillation},
  author={Shadin, Nazmus Shakib and Zhang, Xinyue and Wang, Jingyi and Pan, Miao},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
