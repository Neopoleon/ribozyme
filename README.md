# Classifying Small RNAs via Structure-Aware Graph Neural Networks

**Authors:** Jeff Liu, Lucas Sosnick, Eli Meyers
**Institution:** Stanford University CS224W (Fall 2025)
**Equal Contribution**

---

## Abstract

Small RNAs (rRNAs, microRNAs, ribozymes, etc.) are essential regulators of gene expression whose function emerges from the coupling of nucleotide sequence and secondary structure. This work presents a graph neural network framework for integrated small RNA functional classification that captures both sequence and structural information. Our best model (GIN) achieves **91.8% test accuracy**, significantly outperforming sequence-only baselines and demonstrating that explicit encoding of RNA secondary structure provides stronger signals for functional classification than sequence alone.

---

## Motivation

RNA function emerges from the interplay of nucleotide sequence and secondary structure, where precise structural architectures position key nucleotides for catalysis or molecular recognition. Despite this interdependence, conventional approaches separate structure prediction from functional analysis:

- **Thermodynamic folding algorithms** operate without functional context
- **Classification methods** rely on sequence homology without structural constraints

This separation contradicts biological reality where structure and function co-evolve as mutually constraining properties.

---

## Approach

### Graph Neural Network Framework

RNA molecules are represented as **2D molecular graphs**:
- **Nodes:** Nucleotides with features encoding:
  - Nucleotide identity (A/U/G/C) - 4D one-hot
  - Structural motif (stem/hairpin/bulge/interior loop/multiloop/external) - 7D one-hot
  - Pseudoknot indicator - 1D binary
  - Positional encoding - 1D normalized [0,1]

- **Edges:** Encode connectivity via:
  - Covalent backbone (sequential nucleotides)
  - Base-pairing interactions (Watson-Crick, wobble pairs, pseudoknots)
  - Edge types encoded as 5D one-hot vectors

Through message-passing operations, GNNs learn features capturing higher-order structural motifs (hairpins, internal loops, multi-junctions) that critically determine RNA function but remain inaccessible to sequence-only methods.

---

## Dataset

- **Source:** bpRNA-1m(90) + Rfam annotations
- **Size:** ~20,000 sequences with functional labels
- **Classes:** 19 RNA functional types including:
  - Cis-regulatory elements (riboswitches, IRES, frameshift elements)
  - Gene-associated RNAs (microRNA, CRISPR RNA, antisense RNA)
  - Catalytic RNAs (ribozymes)
  - Small nucleolar RNAs (CD-box, H/ACA-box, scaRNA)
  - snRNA, tRNA, rRNA, introns
- **Split:** 80/10/10 train/validation/test
- **Preprocessing:** Filtered for >90% sequence dissimilarity

---

## Key Research Questions

1. **Which GNN architecture is best suited for RNA classification?**
2. **How important is secondary structure, and which aspects are most useful?**
3. **How does classification accuracy scale with training data?**

---

## Models Evaluated

### GNN Architectures
- **GIN (Graph Isomorphism Network):** Sum aggregation with 2-layer MLPs
- **GCN (Graph Convolutional Network):** Spectral-style convolution
- **GAT (Graph Attention Network):** Attention-weighted message passing

### Baselines
- Sequence-only GNN variants (linear graph, nucleotide features only)
- Transformer on raw nucleotide sequences

### Architecture Details
- 3 graph convolution layers
- Hidden dimension: 128 (GIN, GCN), 512 (GAT)
- Global mean pooling
- 2-layer MLP classifier
- Batch normalization, ReLU activation, dropout (0.3)

---

## Results

### Main Performance (Test Accuracy)

| Model | Test Accuracy | vs Sequence Baseline |
|-------|---------------|---------------------|
| **GIN (structure-aware)** | **91.8%** | +19.3% |
| GCN (structure-aware) | 85.7% | +39.7% |
| GAT (structure-aware) | 80.8% | +47.0% |
| Transformer (sequence) | 81.4% | - |
| GIN (sequence-only) | 72.5% | baseline |
| GCN (sequence-only) | 46.0% | baseline |
| GAT (sequence-only) | 33.8% | baseline |

**Key Finding:** All structure-aware models substantially outperform sequence-only baselines. Removing structural information reduced GIN accuracy from 91.8% to 72.5%.

### Feature Ablation Study

**Most Important Features:**
1. **Structural annotation** (motif labels): Largest performance drop when removed
2. **Nucleotide sequence**: Essential for disambiguation
3. **Positional encoding**: Surprisingly effective (77.9% accuracy with structure but no nucleotide identity)
4. **Pseudoknot indicators**: Minimal impact

**Insight:** Positional encoding + structural topology alone captures substantial functional signal, suggesting global structural organization is highly informative even without explicit nucleotide identity.

### Data Scaling

- **GIN** consistently outperformed GCN and GAT across all data fractions (10%, 20%, 40%, 80%, 100%)
- **With only 20% training data**, GIN surpassed full-data GAT performance
- Performance saturates beyond 80%, suggesting **model capacity** becomes the limiting factor

### Training Dynamics

- **Structure-aware GIN:** Rapid convergence, stable training, minimal generalization gap
- **Sequence-only baselines:** Slower convergence, lower final accuracy
- **Transformer:** Showed overfitting (96% train accuracy, 80% validation accuracy)

### Per-Class Performance

- Most classes achieved **high F1 scores (>0.85)**
- Weaker performance on low-support classes (class imbalance)
- UMAP visualization showed inter-class confusion for functionally related RNA types (e.g., Gene sRNA)

---

## Conclusions

1. **GIN is the best architecture** for RNA functional classification, achieving 91.8% accuracy
2. **Secondary structure is critical:** Provides stronger signal than sequence alone
3. **Data efficiency:** Structure-aware models learn effectively even with limited data
4. **Structural topology matters:** Long-range base-pairing interactions are the dominant signal

The learned representations integrate sequence composition and structural topology in a unified framework, offering a flexible foundation for RNA biology and synthetic design applications.

---

## Future Directions

### 1. Virtual Nodes
Incorporate higher-order structural abstractions (stems, loops, multi-junctions) as first-class graph entities to improve modeling of long-range dependencies.

### 2. Fine-Grained Functional Labels
Extend beyond broad RNA types to specific properties:
- Ligand specificity for riboswitches
- Catalytic mechanisms for ribozymes
- Target classes for regulatory RNAs

### 3. RNA Design and Generation
Couple functional classifiers with generative models (graph diffusion, structure-conditioned generators) to design synthetic RNAs with specified functional constraints.

---

## Implementation

- **Framework:** PyTorch Geometric
- **Training:** Distributed Data Parallel (DDP) across multiple GPUs
- **Code:** [github.com/Neopoleon/ribozyme](https://github.com/Neopoleon/ribozyme)
- **Optimizer:** Adam
- **Epochs:** 100
- **Batch size:** Distributed across GPUs

---

## References

1. Danaee et al. (2018). bpRNA: large-scale automated annotation and analysis of RNA secondary structure. *Nucleic Acids Research*, 46(11), 5381-5394.

2. Ontiveros-Palacios et al. (2024). Rfam database updates. *Nucleic Acids Research*.

3. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.

4. Veličković et al. (2018). Graph Attention Networks. *ICLR*.

5. Xu et al. (2019). How Powerful are Graph Neural Networks? *ICLR*.

---

**Publication:** Medium (December 2025)
**Article URL:** [lsosnick.medium.com/classifying-small-rnas...](https://lsosnick.medium.com/classifying-small-rnas-via-structure-aware-graph-neural-networks-45998ca3de61)