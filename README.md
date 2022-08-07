# GERMAN-PHI
A novel graph embedding representation learning based on multi-head attention mechanism for predicting phage-host interactions
## Overview
Here we provide a method of GERMAN-PHI for phage-host association prediction. It can achieve a successful performance on sparse and non-connect phage-host association network. The repository is organised as follows:
- `data/` contains the necessary dataset files for experiments;
- `models/` contains the implementation of the GAT network (`gat.py`);
- `pre_trained/` contains a pre-trained Cora model (achieving 84.4% accuracy on the test set);
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
