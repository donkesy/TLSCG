# TLSCG: Transfer Learning-Based Smart Contract Generation to Empower Unknown Vulnerability Detection in Blockchain Services

# Required Packages
- python 3.10
- pytorch 2.0.0+cu118
- pandas 1.3.5
- numpy 2.0.2

The overview of our proposed method TLSCG consists of three modules: 1) Generation Model Pre-Training, 2) Transfer Learning, and 3) Vulnerability Detection.

# Datasets
We use the same dataset as [Qian et al., 2023](https://github.com/Messi-Q/Cross-Modality-Bug-Detection) and [Li et al., 2023](https://github.com/Secbrain/VulHunter).


## Data Processing
The data processing code, located in the data_processing/ directory, is primarily responsible for converting raw smart contracts into a model-compatible opcode sequence representation and generating the multi-source embeddings required by the OpTrans classifier.

We utilize the solc-select tool and the solc compiler (evm version 1.11.6) to compile the contracts into their underlying EVM opcode sequences. The opcode sequence is the primary input for the VAE/GAN generation model and the OpTrans classifier.

# Running

The TLSCG framework consists of three key phases:

## 1. Generation Model Pre-Training
-   A generative model (`baseVAE.py`, `train_baseVAE2.py`) is pre-trained using a large number of **normal contracts**.
-   The loss function includes **ELBO Loss**, **Bigram Loss** , and **Adversarial Loss**.

## 2. Transfer Learning
-   When a new vulnerability type is identified, the pre-trained generation model is fine-tuned using a **small amount** of contracts labeled with that specific vulnerability (`train_baseVAE2.py`).
-   This process quickly produces a specialized generation model capable of synthesizing diverse and realistic contracts for the new vulnerability type.

## 3. Vulnerability Detection
-   A balanced training dataset (`generate_smart_contract.py`) is constructed by mixing **real** contracts and **generated anomalous** contracts.
-   This mixed dataset is used to train the **OpTrans** classification model (`OpTrans.py`), which is capable of detecting both known and potentially unknown vulnerabilities.
-   `unknown_detect.py` contains the logic to verify the model's ability to detect anomalous contracts.




