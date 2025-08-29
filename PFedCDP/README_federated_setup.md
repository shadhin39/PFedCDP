# Federated Learning Setup - 50 and 100 Clients

This document describes how to run the federated learning experiments with 50 and 100 clients.

## Overview

The project has been configured to support federated learning with:
- **50 clients**: Standard federated learning setup
- **100 clients**: Large-scale federated learning setup

## File Structure

### 50 Clients Setup
- `client.py` - CIFAR-10 client (supports partition IDs 0-49)
- `client_cbis_ddsm.py` - CBIS-DDSM dataset client (supports partition IDs 0-49)
- `client_mias.py` - MIAS dataset client (supports partition IDs 0-49)
- `server_cbis_ddsm.py` - Server for CBIS-DDSM (min 10 clients)
- `server_mias.py` - Server for MIAS (min 10 clients)

### 100 Clients Setup
- `client_100.py` - CIFAR-10 client (supports partition IDs 0-99)
- `client_cbis_ddsm_100.py` - CBIS-DDSM dataset client (supports partition IDs 0-99)
- `server_cbis_ddsm_100.py` - Server for CBIS-DDSM (min 20 clients)

## Running 50 Clients Setup

### 1. Start the Server

#### For CBIS-DDSM Dataset:
```bash
python server_cbis_ddsm.py
```

#### For MIAS Dataset:
```bash
python server_mias.py
```

#### For CIFAR-10 Dataset:
```bash
python server.py
```

### 2. Start Clients (50 clients)

Open multiple terminals and run clients with different partition IDs:

#### For CBIS-DDSM:
```bash
# Terminal 1
python client_cbis_ddsm.py --partition-id 0

# Terminal 2
python client_cbis_ddsm.py --partition-id 1

# ... continue for partition IDs 0-49
# Terminal 50
python client_cbis_ddsm.py --partition-id 49
```

#### For MIAS:
```bash
# Terminal 1
python client_mias.py --partition-id 0

# Terminal 2
python client_mias.py --partition-id 1

# ... continue for partition IDs 0-49
```

#### For CIFAR-10:
```bash
# Terminal 1
python client.py --partition-id 0

# Terminal 2
python client.py --partition-id 1

# ... continue for partition IDs 0-49
```

## Running 100 Clients Setup

### 1. Start the Server

#### For CBIS-DDSM Dataset:
```bash
python server_cbis_ddsm_100.py
```

### 2. Start Clients (100 clients)

#### For CBIS-DDSM:
```bash
# Terminal 1
python client_cbis_ddsm_100.py --partition-id 0

# Terminal 2
python client_cbis_ddsm_100.py --partition-id 1

# ... continue for partition IDs 0-99
# Terminal 100
python client_cbis_ddsm_100.py --partition-id 99
```

#### For CIFAR-10:
```bash
# Terminal 1
python client_100.py --partition-id 0

# Terminal 2
python client_100.py --partition-id 1

# ... continue for partition IDs 0-99
```

## Configuration Details

### 50 Clients Setup
- **Minimum clients for training**: 10 (20% of 50)
- **Minimum clients for evaluation**: 10
- **Data partitioning**: Dataset divided into 50 equal partitions
- **Training rounds**: 10 rounds

### 100 Clients Setup
- **Minimum clients for training**: 20 (20% of 100)
- **Minimum clients for evaluation**: 20
- **Data partitioning**: Dataset divided into 100 equal partitions
- **Training rounds**: 20 rounds
- **Fraction fit**: 30% of available clients used for training
- **Fraction evaluate**: 30% of available clients used for evaluation

## Data Partitioning

### CIFAR-10
- Uses Flower's `FederatedDataset` with IID partitioning
- Each client gets an equal portion of the training data
- Test data is shared across all clients

### CBIS-DDSM and MIAS
- Custom partitioning function divides data sequentially
- Each client gets a contiguous slice of the dataset
- Last client gets any remaining samples

## Hardware Requirements

### For 50 Clients:
- Minimum 16GB RAM recommended
- Multi-core CPU (8+ cores recommended)
- GPU optional but recommended for faster training

### For 100 Clients:
- Minimum 32GB RAM recommended
- High-performance multi-core CPU (16+ cores recommended)
- Multiple GPUs recommended for large-scale experiments

## Monitoring

- Server logs show aggregation results after each round
- Client logs show local training progress
- Accuracy and loss metrics are tracked and aggregated
- Model checkpoints are saved periodically (every 5 rounds for 100-client setup)

## Notes

1. **Scalability**: The 100-client setup is designed for research purposes and may require significant computational resources.

2. **Network**: All clients connect to `127.0.0.1:8080` by default. For distributed setups, modify the server address in client files.

3. **Synchronization**: The server waits for the minimum number of clients before starting each round.

4. **Data Distribution**: Current implementation uses IID data distribution. For non-IID experiments, modify the partitioning logic.

5. **Performance**: Training time increases significantly with more clients. Consider using simulation mode for large-scale experiments.