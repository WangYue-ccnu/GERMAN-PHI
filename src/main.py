
import numpy as np
import pandas as pd
from train import train
from random import random
from sklearn.metrics import roc_auc_score
def main(ith):
  train_data = []
  test_data = []
  for i in range(1436):
    r = random()
    if r >= 0.2:
      train_data.append(i)
    else:
      test_data.append(i)
  train_dt = np.array(train_data)
  test_dt = np.array(test_data)
  np.save('data/loss_weight_decay/train{}.npy'.format(str(ith)), train_dt)
  np.save('data/loss_weight_decay/test{}.npy'.format(str(ith)), test_dt)
  print('len(train_data):' + str(len(train_data)))
  print('len(test_data):' + str(len(test_data)))
  test_labels, scores,out_come = train(train_data, test_data)
  print('test_labels:\n')
  print(test_labels)
  print('scores:\n')
  print(scores)
  tes_lbl = np.array(test_labels)
  sc = np.array(scores)
  result_scores = np.zeros((out_come.shape[0], out_come.shape[1]))
  for i in range(out_come.shape[0]):
    for j in range(out_come.shape[1]):
      result_scores[i, j] = out_come[i, j]
  np.savetxt('data/loss_weight_decay/result_scores{}.txt'.format(str(ith)), result_scores)
  np.save('data/loss_weight_decay/scores{}.npy'.format(str(ith)), sc)
  np.save('data/loss_weight_decay/true_label{}.npy'.format(str(ith)), tes_lbl)
  auc = roc_auc_score(test_labels, scores)
  return auc
if __name__ == "__main__":
  auc=[]
  for i in range(300):
    auc.append(main(i))
    print(auc)
  print(auc)
  auc = np.array(auc)
  index_ = [i for i in range(len(auc))]
  auc_df = pd.DataFrame(index=index_, data=auc)
  auc_df.to_csv('data/loss_weight_decay/AUC_300.csv')