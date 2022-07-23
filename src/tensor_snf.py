import numpy as np
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
def FindDominantSet(W,K):
    m, n = W.shape
    # DS = np.zeros((m, n))
    # DS=tf.zeros([m, n]) #tensor 常量
    DS_=tf.Variable(np.zeros(shape=[m, n], dtype=np.float64)) #tf.Variable(tf.zeros([m,n]))
    for i in range(m):
        # index = np.argsort(W[i, :])[-K:] # get the closest K neighbors
        index = tf.argsort(W[i, :])[-K:]  # get the closest K neighbors
        # DS[i, index] = W[i, index]  # keep only the nearest neighbors
        row=[i]*index.shape[0]
        row=tf.to_int32(row)
        column=index
        ss=tf.stack([row,column],axis=0)  #ss=[1,1,1,1],[2,3,4,5]
        indexs=tf.unstack(ss,axis=1)  #indexs=[1,2],[1,3],[1,4],[1,5]
        W_=tf.gather_nd(W,indexs)  #取值W indexs位置上的元素
        DS=tf.scatter_nd_update(DS_,indexs,W_)   #使用W_的值对DS中indexs位置重新赋值
    #normalize by sum
    # B = np.sum(DS, axis=1)
    B_=tf.reduce_sum(DS, axis=1)
    # B=B_.reshape(len(B_),1)
    len_b=B_.shape[0]
    B = tf.reshape(B_,(len_b, 1))
    # DS = DS/B
    DS=tf.div(DS,B)
    return DS
#--------------------------------------------------

def normalized(W,alpha):
    m, n = W.shape
    # W = W+alpha*np.identity(m)
    W=tf.add(W,alpha*tf.ones([m]))
    # return (W+np.transpose(W))/2
    return tf.add(W,tf.transpose(W,[1,0]))/2



def SNF(Wall, K, t, alpha=1):
    C = len(Wall)
    m,n = Wall[0].shape
    print('wall0')
    print(Wall[0])
    print('Wall1')
    print(Wall[1])
    for i in range(C):
        # B = np.sum(Wall[i], axis=1)
        B_=tf.reduce_sum(Wall[i], axis=1)
        len_b = B_.shape[0]
        # B = B.reshape(len_b, 1)
        B = tf.reshape(B_, (len_b, 1))
        # Wall[i] = Wall[i] / B    #按行标准化
        Wall[i]=Wall[i] / B
        # print(Wall[i])
        # Wall[i] = (Wall[i] + np.transpose(Wall[i])) / 2  #成为对称阵
        Wall[i]= tf.add(Wall[i], tf.transpose(Wall[i], [1, 0])) / 2

    newW = []

    for i in range(C):
        newW.append(FindDominantSet(Wall[i], K))
    print(newW)
    # Wsum = np.zeros((m, n))
    Wsum=tf.zeros([m, n])
    # 对两个矩阵求和
    for i in range(C):
        Wsum=tf.cast(Wsum,dtype=tf.float32)
        Wall[i]=tf.cast(Wall[i], dtype=tf.float32)
        Wsum =Wsum + Wall[i]
        print('Wsum:')
        print(Wsum)
        # Wsum += Wall[i]
        # Wsum=tf.add(Wsum,Wall[i])

    for iteration in range(t):
        Wall0 = []
        for i in range(C):
            # temp = np.dot(np.dot(newW[i], (Wsum - Wall[i])), np.transpose(newW[i])) / (C - 1)
            newW[i]=tf.cast(newW[i],dtype=tf.float32)
            temp=tf.matmul(tf.matmul(newW[i], tf.subtract(Wsum,Wall[i])),tf.transpose(newW[i],[1,0]))/(C-1)
            Wall0.append(temp)

        for i in range(C):
            Wall[i] = normalized(Wall0[i], alpha)

        # Wsum = np.zeros((m, n))
        Wsum = tf.zeros([m, n])
        for i in range(C):
            # Wsum += Wall[i]
            Wsum=tf.add(Wsum, Wall[i])



    # W = Wsum / C
    W=tf.div(Wsum,C)
    # B = np.sum(W, axis=1)
    B_=tf.reduce_sum(W,axis=1)
    len_b=B_.shape[0]
    # B = B.reshape(len(B), 1)
    B=tf.reshape(B_, (len_b, 1))
    # W /= B
    W=tf.div(W,B)
    # W = (W + np.transpose(W) + np.identity(m)) / 2
    W=tf.add(W,tf.add(tf.transpose(W,[1,0]),tf.ones([m])))/2
    return W

# m,n=W[0].shape
# print(m,n)
# print(am)
if __name__ == '__main__':
    W = []
    am = np.loadtxt("../prepare/snf/152/QS-152H.txt")  # 输入输出可以改
    bm = np.loadtxt("../prepare/snf/152/gip152-h.txt")
    W.append(am)
    W.append(bm)
    out = "../prepare/snf/152/snf152H.csv"  # 输出
    fused_simDr = SNF(W, K=5, t=3, alpha=1.0)  # 参数可以改
    np.savetxt(out, fused_simDr, fmt='%f', delimiter=",")
    # print(fused_simDr)

