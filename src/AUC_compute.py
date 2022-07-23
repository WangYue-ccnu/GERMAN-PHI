import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
'''
计算当前文件夹下所有scores/true_label 的AUC值
'''
auc=[]
for i in range(223):
    NIMCGAT_scores=np.load('data/loss_weight_decay/scores{}.npy'.format(str(i)))
    NIMCGAT_True_label=np.load('data/loss_weight_decay/true_label{}.npy'.format(str(i)))
    scores=np.array(NIMCGAT_scores)
    true_label=NIMCGAT_True_label
    auc.append(roc_auc_score(true_label,list(scores)))
index_ = [i for i in range(len(auc))]
auc_df = pd.DataFrame(index=index_, data=auc)
auc_df.to_csv('data/loss_weight_decay/AUC_255.csv')
print(auc_df)
