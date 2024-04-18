**Gode**
=====
Source code and data for our paper "Enhancing Molecule Representations with Molecule-Centric Knowledge Graphs".

*(Old Title: "Bi-level Contrastive Learning for Knowledge-Enhanced Molecule Representations")*


**Data Preparation**:
```bash
unzip gode_data.zip
```


Code
===

Our code is in the "gode_code" folder.

Here's an overview of the code of corresponding functions:

Molecule-level Pre-training
---
```bash
/m_level_pretrain_and_fine_tune/
```
KGE training (K-GNN Embedding Initialization)
---
```bash
/kg_emb/
```

KG-level Pre-training
---
```bash
/k_level_pretrain/
```

Contrastive Learning
---
```bash
/contrastive_learning/
```

Fine-tuning
---
```bash
/m_level_pretrain_and_fine_tune/
```




Data
===



We detailed the dataset (MolKG) construction and data process in Appendix B of the paper.

After unzipping it, the folder contains

The Entire MolKG Dataset  
---
```bash
gode_data/data_process/KG_processed.csv
```


Processing Scripts 
---
```bash
gode_data/dataset_construction/
```

Once again, thanks a lot for your patience!
