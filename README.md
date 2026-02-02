# R3J-AGNN
**Geometry-Aware Prediction of RNA Three-Way Junction Inter-Branch Angles from Secondary Structure**

## Introduction

R3J-AGNN is a geometry-aware deep learning framework for **predicting inter-branch angular configurations of RNA three-way junctions (3WJ)** directly from RNA secondary structure information.

Unlike template-based or full-coordinate modeling approaches, R3J-AGNN focuses on estimating **key geometric parameters**—the relative angles between the three helices forming a 3WJ—which play a dominant role in determining the global 3D organization of junction-containing RNAs.

The model takes as input only:

* RNA sequence, and
* RNA secondary structure in dot-bracket notation,

and predicts three inter-helical angles (Θ1​,Θ2​,Θ3) that characterize the spatial arrangement of the three junction branches.

---

## Method Overview

R3J-AGNN adopts a **dual-resolution hierarchical graph neural network architecture**, consisting of:

### 1. Nucleotide-Level Graph (Fine Resolution)

* Nodes represent individual nucleotides.
* Edges encode local connectivity derived from sequence adjacency and base-pairing relationships in the secondary structure.
* This graph captures **fine-grained, nucleotide-resolution structural context** within and around the 3WJ region.

### 2. Junction Topology Graph (Coarse Resolution)

* Nodes represent structural elements such as helices and loops.
* Edges describe the topological organization of branches in the 3WJ.
* This graph encodes the **coarse-grained topology of the junction**, abstracted entirely from secondary structure information.

The two graphs are jointly processed to model the coupling between **local nucleotide interactions** and **global junction topology**, enabling accurate inference of inter-branch angular geometry.

---

## Angle Definition and Output Representation

During training, inter-branch angle labels are **derived from experimentally resolved RNA 3D structures**, where helix axes are computed and pairwise angles between branches are measured.

To ensure numerical stability and continuity for periodic variables, the model predicts each angle using its **sine and cosine components**, which are later converted to degrees during inference.

> **Note:**
> R3J-AGNN does **not** directly predict atomic coordinates. Instead, the predicted angles provide **geometric constraints** that can be used as priors for downstream RNA 3D scaffold reconstruction or structure modeling pipelines.

---

## 🔧 Dependencies

We recommend using a Conda environment.

### Environment

* Python 3.8+
* PyTorch ≥ 1.10
* PyTorch Geometric (compatible with PyTorch/CUDA version)
* BioPython
* NetworkX
* NumPy
* Pandas
* tqdm

```bash
# Install PyTorch (see https://pytorch.org for CUDA-specific instructions)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch_geometric

# Install remaining dependencies
pip install biopython networkx numpy pandas tqdm
```

---

## Directory Structure

```
R3J-AGNN/
├── main.py                 # Main inference entry point
├── model.py                # Dual-graph neural network architecture
├── predict.py              # Data preprocessing and graph construction
├── Train.py                # Model training and cross-validation
├── R3J-AGNN_model.pkl      # Pre-trained model weights
├── example/
│   └── example.fasta       # Example input file
└── Datasets/               # (Optional) Training data directory
    ├── TrainingSet/
    └── TreeGraph/
```

---

## 🚀 Usage

### 1. Input Format

R3J-AGNN expects a **3-line FASTA-like format**:

```
>RNA_ID
AUGGCU...
(((...)))
```

* Line 1: RNA identifier
* Line 2: RNA sequence (A, U, G, C)
* Line 3: Secondary structure (dot-bracket notation)

---

### 2. Run Prediction

```bash
python main.py \
  --input example/example.fasta \
  --model R3J-AGNN_model.pkl \
  --device cuda
```

**Arguments:**

* `--input`: Path to input FASTA file
* `--model`: Path to trained model checkpoint (.pkl)
* `--device`: `cuda` or `cpu` (default: auto-detect)
* `--output`: (Optional) Output file path

---

### 3. Output Interpretation

The program identifies three-way junction nodes and reports the predicted inter-branch angles (in degrees).

---

## 📤 Example Output

```
[*] Loading model from R3J-AGNN_model.pkl...
[+] Model loaded successfully.
[+] Loaded 1 RNA entries.
[*] Running inference...

==========================================================================================
RNA Name | NodeIdx | Angle Theta1 | Angle Theta2 | Angle Theta3
------------------------------------------------------------------------------------------
example  |    1    |   120.13°    |   125.14°    |   114.22°
==========================================================================================
```

---

## 🔧 Model Training

To retrain or fine-tune the model:

1. Prepare training data as PyTorch graph objects (`.pt` files).
2. Place them in `Datasets/TrainingSet/`.
3. Edit `Train.py`:

```python
DATA_PATH = "/path/to/Datasets/TrainingSet/trainData.pt"
```

The training script performs **5-fold cross-validation** and saves the best-performing model for each fold.

---

## 📄 Citation

(Coming soon)

---

## 📬 Contact

**Ya-Zhou Shi or Ya-Lan Tan**
*School of Mathematics & Statistics
*Wuhan Textile University
*📧 [yzshi@wtu.edu.cn](mailto:yzshi@wtu.edu.cn) or yltan@wtu.edu.cn

---
