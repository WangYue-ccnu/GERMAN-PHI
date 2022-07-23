# import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def masked_accuracy(preds, labels, mask, negative_mask):  #损失函数修改处
    """Accuracy with masking."""
    alpha=0.4
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)  #负样本的标签值为0，正样本的标签值为1   #求平方
    #mask += negative_mask   #计算了正样本误差和负样本的误差求和
    # mask = tf.cast(mask, dtype=tf.float32)
    # error *= mask  #*代表相同位置相乘 计算误差时需要将负样本的累加 计算时 计算了正样本中的误差值和负样本中的误差值
    mask = tf.cast(mask, dtype=tf.float32)
    negative_mask = tf.cast(negative_mask, dtype=tf.float32)
    mask_=(1-alpha)*mask+alpha*negative_mask
    mask_=tf.cast(mask_,dtype=tf.float32)
    error*=mask_
    return tf.reduce_sum(error)
    # return tf.sqrt(tf.reduce_mean(error)) #返回所有误差均值的平方

def ROC(outs, labels, test_arr, label_neg):
    scores=[]
    for i in range(len(test_arr)):
        l=test_arr[i]
        scores.append(outs[int(labels[l,0]-1),int(labels[l,1]-1)])
    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i,0]),int(label_neg[i,1])])
    test_labels=np.ones((len(test_arr),1))
    temp=np.zeros((label_neg.shape[0],1))
    test_labels1=np.vstack((test_labels,temp))
    test_labels1=np.array(test_labels1,dtype=np.bool).reshape([-1,1])
    return test_labels1,scores