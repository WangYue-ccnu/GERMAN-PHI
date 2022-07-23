# import tensorflow as tf
from inits import glorot
import tensorflow.compat.v1 as tf
import snf
tf.disable_v2_behavior()


conv1d = tf.layers.conv1d
        
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
  with tf.name_scope('my_attn'):
    if in_drop != 0.0:
       seq = tf.nn.dropout(seq, 1.0 - in_drop)  #概率为1-in_drop的元素被设置为0，其他元素设置为1.0 / in_drop
    seq_fts = seq
    latent_factor_size = out_sz #latent_factor_size=out_sz
    
    # w_1 = glorot([seq_fts.shape[2].value,latent_factor_size])
    #写法应为：w_1 = glorot([seq_fts.shape[2],latent_factor_size]) W1:特征数*转换大小(论文中矩阵乘法写错了，对应论文中Wa的l*r)
    w_2 = glorot([2*seq_fts.shape[2].value,latent_factor_size])
    w_3= glorot([seq_fts.shape[2].value,latent_factor_size])
    # 写法应为：w_2 = glorot([2*seq_fts.shape[2],latent_factor_size]) W2:特征数*转换大小(论文中矩阵乘法写错了，对应论文中Wa的(l+l)*r)
    # w1对应wa，w2对应wb
    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])#注意力分数矩阵E eij
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits[0]) + bias_mat[0]) #归一化便到了注意力的权重矩阵
    if coef_drop != 0.0:
       coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
    if in_drop != 0.0:
       seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
    
    neigh_embs = tf.matmul(coefs, seq_fts[0])
    
    # neigh_embs_aggre_1 = tf.matmul(tf.add(seq_fts[0],neigh_embs),w_1)
    neigh_embs_aggre_2 = tf.matmul(tf.concat([seq_fts[0],neigh_embs],axis=-1),w_2)
    neigh_repre = [seq_fts[0], neigh_embs]
    neigh_embs_snf = snf.SNF(neigh_repre, K=5, t=3, alpha=1.0)
    neigh_embs_aggre_3 = tf.matmul(neigh_embs_snf, w_3)
    # final_embs = activation(neigh_embs_aggre_1) + activation(neigh_embs_aggre_2) + activation(neigh_embs_aggre_3)
    final_embs = activation(neigh_embs_aggre_2) + activation(neigh_embs_aggre_3)
    return final_embs, coefs