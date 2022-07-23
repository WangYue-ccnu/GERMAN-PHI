# import tensorflow as tf
import tensorflow.compat.v1 as tf

from utils import layers
from models.base_gattn import BaseGAttN
from inits import glorot
from metrics import masked_accuracy
tf.disable_v2_behavior()

class GAT(BaseGAttN):
    #model.encoder(feature_in, nb_nodes, is_train,attn_drop, ffd_drop,bias_mat=bias_in,hid_units=hid_units[8], n_heads=n_heads[4,1],residual=residual, activation=nonlinearity)
    def encoder(inputs, nb_nodes, training, attn_drop, ffd_drop,    
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):   #n_heads[0]=4
            attn_temp, coefs = layers.attn_head(inputs, bias_mat=bias_mat,      
                out_sz=hid_units[0], activation=activation,   
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            inputs = attn_temp[tf.newaxis] #增加一个维度
            attns.append(attn_temp) #将每个头得到的表示加入到attns中
        h_1 = tf.concat(attns, axis=-1)   #将4个注意力头拼接 按列拼接
        return h_1, coefs   
            
    def decoder(embed, nd):
        embed_size = embed.shape[1].value  #embed=(nd+nm)*Kr embed_size=Kr
        # 应该为：embed_size = embed.shape[1]
        # with tf.compat.v1.variable_scope("deco"):
        # with tf.compat.v1.variable_scope('w3'):
        #         weight3 = glorot([embed_size,embed_size],'weight3') #Kr*Kr
        with tf.compat.v1.variable_scope('wx1'):
                Wx_1=glorot([embed_size,256],name='wx_1')
        with tf.compat.v1.variable_scope('wx2'):
                Wx_2=glorot([256,128],name='wx_2')
        with tf.compat.v1.variable_scope('wx3'):
                Wx_3=glorot([128,64])
        with tf.compat.v1.variable_scope('bx1'):
             bx_1=glorot([1,256])
        with tf.compat.v1.variable_scope('bx2'):
             bx_2=glorot([1,128])
        with tf.compat.v1.variable_scope('bx3'):
             bx_3=glorot([1,64])

        with tf.compat.v1.variable_scope('wy1'):
             Wy_1 = glorot([embed_size, 256])
        with tf.compat.v1.variable_scope('wy2'):
             Wy_2 = glorot([256, 128])
        with tf.compat.v1.variable_scope('wy3'):
             Wy_3 = glorot([128, 64])
        with tf.compat.v1.variable_scope('by1'):
             by_1 = glorot([1, 256])
        with tf.compat.v1.variable_scope('by2'):
             by_2 = glorot([1, 128])
        with tf.compat.v1.variable_scope('by3'):
             by_3 = glorot([1, 64])
        U=embed[0:nd,:] #疾病的节点表示 nd*Kr  病毒
        V=embed[nd:,:] #microbe的节点表示 nm*Kr  宿主
        X1_=tf.add(tf.matmul(U,Wx_1),bx_1)
        X1=tf.nn.relu(X1_)
        X2_=tf.add(tf.matmul(X1,Wx_2),bx_2)
        X2=tf.nn.relu(X2_)
        X3_=tf.add(tf.matmul(X2,Wx_3),bx_3)
        X=tf.nn.relu(X3_)

        Y1_ = tf.add(tf.matmul(V, Wy_1), by_1)
        Y1 = tf.nn.relu(Y1_)
        Y2_ = tf.add(tf.matmul(Y1, Wy_2), by_2)
        Y2 = tf.nn.relu(Y2_)
        Y3_ = tf.add(tf.matmul(Y2, Wy_3), by_3)
        Y = tf.nn.relu(Y3_)

        # logits=tf.matmul(tf.matmul(U,weight3),tf.transpose(V)) # weight3等价于论文中Wd*Wm^T
        logits=tf.matmul(X,tf.transpose(Y))
        logits=tf.reshape(logits,[-1,1])#转换为一列
        return tf.nn.relu(logits) #将大于零的元素保持不变，小于零的元素置0
    
    def loss_sum(scores, lbl_in, msk_in, neg_msk, weight_decay, coefs, emb):
        loss_basic = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        # para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope='w3')
        # loss_basic += weight_decay * tf.nn.l2_loss(para_decode)
        scope_=['wx1','wx2','wx3','wy1','wy2','wy3','bx1','bx2','bx3','by1','by2','by3']
        for item in scope_:
            para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=item)
            # para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope='weight3')
            loss_basic +=  weight_decay * tf.nn.l2_loss(para_decode)
        return loss_basic

    
    
    
    
    
    
    
    
    
    
     