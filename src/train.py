import time
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from models import GAT
from inits import adj_to_bias
from inits import test_negative_sample
from inits import load_data
from inits import generate_mask
from metrics import masked_accuracy
from metrics import ROC

tf.disable_v2_behavior()
def train(train_arr, test_arr):
    
    # training params
    batch_size = 1 #训练集大小 一次训练所选取的样本数。Batch Size=1，梯度变来变去，非常不准确，网络很难收敛。
    nb_epochs = 300 #训练次数
    lr = 0.005  #学习率
    l2_coef = 0.0005  #l2逻辑回归中的权重系数
    weight_decay = 1e-7 #delay factor γ
    hid_units = [8]  #number of neuronsr
    n_heads = [4, 1] #K
    residual = False
    nonlinearity = tf.nn.elu #激活函数
    model = GAT

    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))
    #interaction (nd+nm)*(nd+nm)  #featrure (nd+nm)*(nd+nm)
    interaction, features, y_train, y_test, train_mask, test_mask, labels = load_data(train_arr, test_arr)
    # 对应论文中的矩阵A,interaction矩阵
    # 对应论文中矩阵X，特征矩阵feature
    # 恢复train矩阵0 1 矩阵 并将矩阵转换为一列
    # 将logist_train/test转换为mask: true false
    # labels关联信息 disease microbe 1
    nb_nodes = features.shape[0]  #节点数 nm+nd
    ft_size = features.shape[1]  #特征数 初始状态：nm+nd

    features = features[np.newaxis] #多增加一个维度 eg：从[[]]变为[[[]]]
    interaction = interaction[np.newaxis]#多增加一个维度 eg：从[[]]变为[[[]]]
    biases = adj_to_bias(interaction, [nb_nodes], nhood=1)
    #bias=adj_to_bias([interaction], [nd+nm], nhood=1)
    
    nd = np.max(labels[:,0]) #疾病数  病毒
    nm = np.max(labels[:,1])#miRNA数  宿主
    nd = nd.astype(np.int32)
    nm = nm.astype(np.int32)
    entry_size = nd * nm
    with tf.Graph().as_default():
        with tf.name_scope('input'):
              feature_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size)) #(1,nb_nodes,ft_size)
              bias_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes)) #(1,nb_nodes,nb_nodes)
              lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size)) #(nd*nm,1)
              msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))#(nd*nm,1)
              neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size,batch_size))#(nd*nm,1) 1列
              attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
              ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
              is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=()) #一个bool数值
        
        final_embedding, coefs = model.encoder(feature_in, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
        scores = model.decoder(final_embedding, nd)
        loss = model.loss_sum(scores, lbl_in, msk_in, neg_msk, weight_decay, coefs, final_embedding)
    
        accuracy = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        
        train_op = model.training(loss, lr, l2_coef)

        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

        with tf.compat.v1.Session() as sess:
          sess.run(init_op)

          train_loss_avg = 0
          train_acc_avg = 0

          for epoch in range(nb_epochs): #训练200次
              
              t = time.time()
              
              ##########    train     ##############
              
              tr_step = 0
              tr_size = features.shape[0] #tr_size=1
              
              neg_mask, label_neg = generate_mask(labels, len(train_arr))
              
              while tr_step * batch_size < tr_size:  #循环一次
                      _,loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                      feed_dict={
                           feature_in: features[tr_step*batch_size:(tr_step+1)*batch_size],  #feature_in =feature[0]
                           bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size], #biases[0]
                           lbl_in: y_train,
                           msk_in: train_mask,
                           neg_msk: neg_mask,
                           is_train: True,
                           attn_drop: 0.1, ffd_drop: 0.1})
                      train_loss_avg += loss_value_tr
                      train_acc_avg += acc_tr
                      tr_step += 1
              print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr,acc_tr, time.time()-t))
          saver = tf.train.Saver()
          saver.save(sess, 'Model_changeable/GAT_model')
          print("Finish traing.")
          
          ###########     test      ############
          
          ts_size = features.shape[0]
          ts_step = 0
          ts_loss = 0.0
          ts_acc = 0.0
    
          print("Start to test")
          while ts_step * batch_size < ts_size:
              out_come, emb, coef, loss_value_ts, acc_ts = sess.run([scores, final_embedding, coefs, loss, accuracy],
                      feed_dict={
                          feature_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                          bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                          lbl_in: y_test,
                          msk_in: test_mask,
                          neg_msk: neg_mask,
                          is_train: False,
                          attn_drop: 0.0, ffd_drop: 0.0})
              ts_loss += loss_value_ts
              ts_acc += acc_ts
              ts_step += 1
          print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
          print('train_emb:\n')
          print(emb)
          out_come = out_come.reshape((nd,nm))
          test_negative_samples = test_negative_sample(labels,len(test_arr),neg_mask.reshape((nd,nm)))
          test_labels, score = ROC(out_come,labels, test_arr,test_negative_samples)

          return test_labels, score,out_come
          sess.close()
