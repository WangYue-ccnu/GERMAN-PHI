

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import pandas as pd
from GIPsimilarity import Gussian_similarity
# import tensorflow as tf
from collections import defaultdict
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def adj_to_bias(adj, sizes, nhood=1):       
    nb_graphs = adj.shape[0]  #图的个数1
    mt = np.empty(adj.shape) #不是空数组，元素为随机值，和adj同大小的矩阵
    # print(sizes[0])
    for g in range(nb_graphs):#一次循环
        mt[g] = np.eye(adj.shape[1]) #对角线为1，其余为0
        for _ in range(nhood):#一次循环
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]): #等价于 sizes[0] nm+nd
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)                            
            
def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten() #按行降维一个维度

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def  load_data(train_arr, test_arr):
    
    labels = np.loadtxt("data/mydata/adj.txt")
    #关联信息 disease microbe 1
    nd = np.max(labels[:,0])
    nm = np.max(labels[:,1])
    nd = nd.astype(np.int32)
    nm = nm.astype(np.int32)
    
    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(nd,nm)).toarray()
    #恢复test矩阵0 1 矩阵
    logits_test = logits_test.reshape([-1,1])
    # 将上述矩阵转换为一列

    logits_train_matrix = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(nd,nm)).toarray()
    #恢复train矩阵0 1 矩阵
    logits_train = logits_train_matrix.reshape([-1,1])
    #将上述矩阵转换为一列

      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])
    #将logist_train/test转换为true false
    
    # M = sio.loadmat('data/HMDAD/interaction.mat')
    # M = M['interaction']
    M=pd.read_csv('data/mydata/p_h_values.csv',header=None)
    M=M.values

    interaction = np.vstack((np.hstack((np.zeros(shape=(nd,nd),dtype=int),M)),np.hstack((M.transpose(),np.zeros(shape=(nm,nm),dtype=int)))))      
    #对应论文中的矩阵A,interaction矩阵
    # F1 = np.loadtxt("data/HMDAD/disease_features.txt")
    # F2 = np.loadtxt("data/HMDAD/microbe_features.txt")
    F1=pd.read_csv('data/mydata/VS_d2star_values.csv',header=None)
    F1=F1.values

    ##用GIP补充HS_d2star缺失部分
    F2_GIP=Gussian_similarity(logits_train_matrix.T)
    F2_d2star=pd.read_csv('data/mydata/HS_d2star_values.csv',header=None)
    F2_d2star=F2_d2star.values
    F2=F2_d2star
    for i in range(F2_d2star.shape[0]):
        for j in range(F2_d2star.shape[1]):
            if F2_d2star[i, j] == 0:
                F2[i,j]=F2_GIP[i,j]
    features = np.vstack((np.hstack((F1,np.zeros(shape=(F1.shape[0],F2.shape[1]),dtype=int))), np.hstack((np.zeros(shape=(F2.shape[0],F1.shape[0]),dtype=int), F2))))
    #对应论文中矩阵X，特征矩阵feature
    features = normalize_features(features)
    return interaction, features, sparse_matrix(logits_train), logits_test, train_mask, test_mask, labels
    #sparse_matrix(logits_train)将矩阵中的0 替换为0.001
def generate_mask(labels,N):  
    num = 0
    
    nd = np.max(labels[:,0])
    nm = np.max(labels[:,1])
    nd = nd.astype(np.int32)
    nm = nm.astype(np.int32)
    
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(nd,nm)).toarray()
    mask = np.zeros(A.shape)
    label_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,nd-1) 
        b = random.randint(0,nm-1) 
        if A[a,b] != 1 and mask[a,b] != 1: #从所有未知样本中 随机选取训练集中正例样本等量的未知样本作为负样本
            mask[a,b] = 1  # 设置负样本，在负样本矩阵中，负样本设置为1
            label_neg[num,0]=a
            label_neg[num,1]=b  #标记负样本的索引
            num += 1
    mask = np.reshape(mask,[-1,1])  
    return mask,label_neg

def test_negative_sample(labels,N,negative_mask): 
    num = 0
    (nd,nm)=negative_mask.shape
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(nd,nm)).toarray() 
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,nd-1) #随机产生与测试集中正例数等量的负例样本
        b = random.randint(0,nm-1) 
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def  glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
def maxpooling(a):
    a=tf.cast(a,dtype=tf.float32)
    b=tf.reduce_max(a,axis=1,keepdims=True)
    c=tf.equal(a,b)
    mask=tf.cast(c,dtype=tf.float32)
    final=tf.multiply(a,mask)
    ones=tf.ones_like(a)
    zeros=tf.zeros_like(a)
    final=tf.where(final>0.0,ones,zeros)
    return final

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized
    
def sparse_matrix(matrix):#把0换成0.001
    sigma = 0.001
    matrix = matrix.astype(np.int32)
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        if matrix[i,0]==0:
           result[i,0]=sigma
        else:
           result[i,0]=1
    return result        
