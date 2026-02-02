# R3J-AGNN
Geometry-Aware Prediction of RNA Three-Way Junctions Using Dual-Graph Neural Networks
## Introduction
----------------
R3J-AGNN is a deep learning framework designed to predict the 3D geometry of RNA three-way junctions (3WJ). Unlike traditional methods that rely on tertiary structure templates, R3J-AGNN utilizes a Dual-Graph Neural Network architecture that combines:
Nucleotide-Graph: Capturing atomic/nucleotide-level interactions (sequence and secondary structure).Tree-Graph: Capturing the high-level topological relationships between helices and loops.By taking only the RNA sequence and secondary structure (dot-bracket) as input, the model predicts the inter-helical angles ($\Theta1, \Theta2, \Theta3$) defined by the Euler angles or helix-to-helix vectors, outputting sine and cosine values to ensure geometric continuity.
## 🔧 Dependencies
-------------------
Before running R3J-AGNN, ensure you have the following dependencies installed. We recommend using a Conda environment.
## 🖥 Environment
-----------------
* Python 3.8+

* PyTorch (>= 1.10)

* PyTorch Geometric (matching PyTorch/CUDA version)

* BioPython

* NetworkX

* NumPy

* Pandas

# Install PyTorch first (check https://pytorch.org/ for your specific CUDA version)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch_geometric

# Install other dependencies
pip install biopython networkx numpy pandas tqdm

## Directory Structure
Ensure your project directory is organized as follows for the scripts to work correctly:
R3J-AGNN/
├── main.py               # Main entry point for prediction
├── model.py              # DualGraphRNAModel architecture definition
├── predict.py            # Data preprocessing and feature engineering logic
├── Train.py              # Script for training/retraining the model
├── R3J-AGNN_model.pkl    # Pre-trained model weights
├── example/
│   └── example.fasta     # Input file example
└── Datasets/             # (Optional) For training data
    ├── TrainingSet/
    └── TreeGraph/
## 🚀 Usage
-----------------
1. Prepare Input Data
R3J-AGNN accepts input in a specific 3-line FASTA format.
Line 1: Header (starts with >)
Line 2: RNA Sequence (A, U, G, C)
Line 3: Secondary Structure (Dot-bracket notation)
2. Run Prediction
Use main.py to predict the junction geometry.
* Command:
* python main.py --input example/example.fasta --model R3J-AGNN_model.pkl
Arguments:
* --input: Path to the input FASTA file containing sequence and structure.
* --model: Path to the trained model checkpoint (.pkl file).
* --device: Compute device (default: cuda if available, else cpu).
* --output: (Optional) Path to save results to a file.
3. Understanding the Output
The program filters for Multi-loop (Junction) nodes and outputs the predicted inter-helical angles in degrees.
## 📤 Output
-----------------
[*] Loading model from R3J-AGNN_model.pkl...
[+] Model loaded successfully.
[+] Loaded 1 RNA entries.
[*] Running inference...

==========================================================================================
RNA Name                 | NodeIdx  | Angle Theta1  | Angle Theta2  | Angle Theta3 
------------------------------------------------------------------------------------------
example                  | 1        |     120.13°   |     125.14°   |     114.22°
==========================================================================================

## 🔧 Model Training
If you wish to retrain the model or fine-tune it on new data:
Prepare your dataset as a PyTorch Graph list (.pt format) and place it in Datasets/TrainingSet/.
Open Train.py and modify the DATA_PATH variable:
* DATA_PATH = "/path/to/your/Datasets/TrainingSet/trainData.pt"
The script performs 5-Fold Cross-Validation and saves the best model for each fold in the output directory.

## 📄 Citation


## 📬 Contact
For questions, bug reports, or contributions, please contact:
Ya-Zhou Shi
School of Mathematics & Statistics, Wuhan Textile University
📧 yzshi@wtu.edu.cn
