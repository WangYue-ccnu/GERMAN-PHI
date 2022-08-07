# GERMAN-PHI
A novel graph embedding representation learning based on multi-head attention mechanism for predicting phage-host interactions
## Overview
Here we provide a method of GERMAN-PHI for phage-host association prediction. It can achieve a successful performance on sparse and non-connect phage-host association network. The repository is organised as follows:
- `src/data/mydata` contains the necessary dataset files for experiments;
- `src/models/` contains the implementation of the GAT network and NIMC decoder (`gat.py`);
- `src/utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
- `src` contains:
    For fusing the node representation and it's self-star-network representation (`snf.py`)
## Dependencies

The script has been tested running under Python 3.8.2, with the following packages installed (along with their dependencies):

- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow==1.6.0`
## Reference
If you make advantage of the GERMAN-PHI model in your research, please cite the following in your manuscript:

```
@article{
  title="{A novel graph embedding representation learning with multi-head attention mechanism for predicting phage-host interactions}",
  author={Y, Wang and H, Sun and H, Wang and D, Li and W, Zhao and X,Jiang and X, Shen},
  year={2022},
  note={under review},
}

@article{Long(2021),
	title = {Predicting human microbe-disease associations via graph attention networks with inductive matrix completion},
	volume = {22},
	journal = {Briefings in Bioinformatics},
	author = {Y, Long and J, Luo and Y et al., Zhang},
	year = {2021},
	pages = {bbaa146},
}

@article{Li(2020),
	title = {Neural inductive matrix completion with graph convolutional networks for {miRNA}-disease association prediction},
	volume = {36},
	journal = {Bioinformatics},
	author = {J, Li and S, Zhang and T et al., Liu },
	year = {2020},
	pages = {2538--2546},
}
```

You may also be interested in the following unofficial ports of the GAT model:
- \[Keras\] [keras-gat](https://github.com/danielegrattarola/keras-gat), currently under development by [Daniele Grattarola](https://github.com/danielegrattarola);
- \[PyTorch\] [pyGAT](https://github.com/Diego999/pyGAT), currently under development by [Diego Antognini](https://github.com/Diego999).
