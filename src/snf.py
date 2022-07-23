import numpy as np
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
def FindDominantSet(W,K):
    m, n = W.shape
    DS_=tf.Variable(tf.zeros([m,n]),dtype=tf.float32) #tf.Variable(tf.zeros([m,n]))
    for i in range(m):
        index = tf.argsort(W[i, :])[-K:]  # get the closest K neighbors
        row=[i]*index.shape[0]
        row=tf.to_int32(row)
        column=index
        ss=tf.stack([row,column],axis=0)  #ss=[1,1,1,1],[2,3,4,5]
        indexs=tf.unstack(ss,axis=1)  #indexs=[1,2],[1,3],[1,4],[1,5]
        W_=tf.gather_nd(W,indexs)  #取值W indexs位置上的元素
        W_=tf.cast(W_,dtype=tf.float32)
        DS=tf.scatter_nd_update(DS_,indexs,W_)   #使用W_的值对DS中indexs位置重新赋值
    B_=tf.reduce_sum(DS, axis=1)  #1*DS.shape[1]
    len_b=B_.shape[0]
    #将B_中的0换为1
    B_=tf.cast(B_,dtype=tf.float32)
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(B_, zero)
    one = tf.ones(B_.shape)
    B_ = tf.where(where, B_, one)
    B = tf.reshape(B_, (len_b, 1))
    DS = DS/B

    return DS
#--------------------------------------------------

def normalized(W,alpha):
    m, n = W.shape
    W=tf.add(W,alpha*tf.ones([n]))
    return W



def SNF(Wall, K, t, alpha=1):
    C = len(Wall)
    m,n = Wall[0].shape
    for i in range(C):
        B_=tf.reduce_sum(Wall[i], axis=1)
        B_=tf.cast(B_,dtype=tf.float32)
        len_b = B_.shape[0]
        # 将B_中的0换为1
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(B_, zero)
        one=tf.ones(B_.shape)
        B_=tf.where(where,B_,one)
        B = tf.reshape(B_, (len_b, 1))
        # Wall[i] = Wall[i] / B    #按行标准化
        Wall[i]=tf.cast(Wall[i],dtype=tf.float32)
        Wall[i]=Wall[i] / B
    newW = []

    for i in range(C):
        newW.append(FindDominantSet(Wall[i], K))
    Wsum=tf.zeros([m, n])
    # 对两个矩阵求和
    for i in range(C):
        Wsum=tf.cast(Wsum,dtype=tf.float32)
        Wall[i]=tf.cast(Wall[i], dtype=tf.float32)
        Wsum =Wsum + Wall[i]

    for iteration in range(t):
        Wall0 = []
        for i in range(C):
            # temp = np.dot(np.dot(newW[i], (Wsum - Wall[i])), np.transpose(newW[i])) / (C - 1)
            newW[i]=tf.cast(newW[i],dtype=tf.float32)
            temp=tf.matmul(tf.matmul(newW[i],tf.transpose(tf.subtract(Wsum,Wall[i]),[1,0])),newW[i])/(C-1)
            Wall0.append(temp)

        for i in range(C):
            Wall[i] = normalized(Wall0[i], alpha)

        Wsum = tf.zeros([m, n])
        for i in range(C):
            Wsum=tf.add(Wsum, Wall[i])

    W=tf.div(Wsum,C)
    B_=tf.reduce_sum(W,axis=1)
    len_b=B_.shape[0]
    # 将B_中的0换为1
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(B_, zero)
    one = tf.ones(B_.shape)
    B_ = tf.where(where, B_, one)
    B=tf.reshape(B_, (len_b, 1))
    W = W/B

    return W

