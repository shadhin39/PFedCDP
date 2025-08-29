## ✨ Key Features

* **Heterogeneity-Aware Client Clustering**: Clients are stratified into clusters (e.g., High, Medium, Low) based on their computational resources (CPU, memory, battery, network). Each cluster is assigned a neural network architecture tailored to its capabilities, preventing bottlenecks from resource-constrained devices.
* **Fisher Information-Guided Personalization**: Employs Fisher Information to identify and preserve crucial local model parameters for personalization, while integrating generalized knowledge from the global model for less critical parameters.
* **Knowledge Distillation and Multi-teacher Knowledge Distillation**:
    * **Client-Side**: A powerful server-side classification layer acts as a "teacher" to guide the training of each client's personalized classifier.
    * **Server-Side**: A post-hoc refinement phase uses the cluster-aggregated models as an ensemble of teachers to train a final, robust student model.
* **Shapley Value-Weighted Aggregation**: In the post-hoc stage, Shapley values are used to compute the feature importance of each teacher model, providing a principled method for weighting their contributions during knowledge distillation.
* **Supervised Contrastive Learning (SCL)**: SCL is used during server-side refinement to learn more discriminative representations by promoting intra-class compactness and inter-class separability on a public dataset.
* **Privacy and Communication Efficiency**: Integrates **Differential Privacy** by adding calibrated noise to updates and **Quantization** to reduce the model update size, ensuring user privacy and reducing communication overhead.

---
## ⚙️ Framework Architecture

The PFedCDP framework operates in two main phases: an iterative personalized training loop and a final server-side refinement stage.

![image alt](https://github.com/shadhin39/PFedCDP/blob/main/Group%20166.png?raw=true)
> **Figure (a)**: High-level overview of the PFedCDP architecture with heterogeneous client clusters and the server's post-hoc refinement stage. 

![image alt](https://github.com/shadhin39/PFedCDP/blob/main/Group%20167.png?raw=true)
>  **Figure (b)**: A detailed view of the iterative personalized training loop, highlighting model decomposition, Fisher Information-based personalization, and knowledge distillation.


---
## 📦 Datasets

Our experiments simulate a realistic medical imaging scenario using two public mammography datasets.

* **Private Client Data (non-IID)**: The **CBIS-DDSM** dataset is used to simulate private data held by clients. To model statistical heterogeneity, the training data is partitioned among clients using a **Dirichlet distribution ($\alpha=0.5$)**, ensuring a non-IID data distribution that mirrors real-world scenarios.
* **Public Server Data**: The **MIAS** dataset serves as the public, auxiliary data on the server. It is used exclusively during the post-hoc refinement phase to enhance the final model's generalization capabilities without accessing any private client data.

---
## 🔧 Dependencies

The simulation is built using the Flower framework. The following dependencies are required:

* Python >= 3.10
* TensorFlow >= 2.15 or PyTorch >= 2.0
* flwr (`Flower`)
* NumPy
* Shap
* Pandas
* Matplotlib

---
## ⚙️ Parameters

The following are the key parameters used in the experimental setup.

| Argument | Description | Value |
| :--- | :--- | :--- |
| `federated_rounds` | Number of federated training rounds. | 20 |
| `local_epochs` | Number of local training epochs per client per round. | 20 |
| `alpha` | Controls the degree of non-IID-ness (Dirichlet dist.). | 0.5 |
| `epsilon` | Privacy budget for Differential Privacy. | 5.0 |
| `C2` | $L_2$ norm clipping threshold for Differential Privacy. | 2.5 |
| `quantization_bits` | Number of bits for model update quantization. | 8 |
| `T_KD` | Temperature parameter for knowledge distillation. | Varies |
| `lambda` | Hyperparameter to balance $L_{CE}$ and $L_{KD}$ losses. | Varies |

---
## 🚀 Usage

The simulation is managed by the Flower framework over a gRPC-based architecture.

1.  **Start the Server- PFedCDP**: The server orchestrates the federated learning process, including model distribution, aggregation, and post-hoc refinement.
    ```bash
    python server_main_contrastive_learning.py
    ```
2.  **Start the Clients- PFedCDP**: Launch clients for each of the three capability clusters. Each cluster listens on a dedicated port (e.g., 8080, 8081, 8082). Open new terminals for each client command.

    * **High Capability Client (Cluster 1):**
        ```bash
        python client.py --port=8080
        ```
    * **Medium Capability Client (Cluster 2):**
        ```bash
        python client1.py --port=8081
        ```
    * **Low Capability Client (Cluster 3):**
        ```bash
        python client2.py --port=8082
        ```
        
 3. **Start the Server- FedAvg**: The server orchestrates the federated learning process.
    ```bash
    python server.py
    ```
4. **Start the Clients- FedAvg**: Open new terminals for each client command (minimum number of clients required = 2). 

    * **Client 1:**
        ```bash
        python client.py
        ```
    * **Client 2:**
        ```bash
        python client.py
        ```
---
## 📊 Results

PFedCDP significantly outperforms the standard FedAvg baseline and approaches the performance of a centralized model, demonstrating its effectiveness in handling heterogeneity and privacy constraints.

| Method | CBIS-DDSM (Private Test Acc. %) | MIAS (Public Test Acc. %) |
| :--- | :---: | :---: |
| Centralized (Upper Bound) | 96.2 | 99.8 |
| Standard FedAvg | 85.7 | 91.3 |
| **PFedCDP (Ours)** | **94.1** | **99.5** |
> Table: Final test accuracy comparison of PFedCDP against baselines.

---
