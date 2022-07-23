import tensorflow.compat.v1 as tf
import snf
import numpy as np
tf.enable_eager_execution()

# ref = tf.Variable(np.zeros(shape=[6, 6], dtype=np.float32))
# for i in range(6):
#     updates = tf.constant([i+1, i+2, i+3, i+4], dtype=tf.float32)
#     indices = tf.constant([[i, 4], [i, 0], [i, 5], [i, 0]], dtype=tf.int32)
#     DS = tf.scatter_nd_update(ref, indices, updates)
#     print(DS)

W1=tf.constant([1, 2, 3, 4, 5, 6,0,0,0],shape=[3,3])
W2=tf.constant([0, 0, 0, 5, 6,7,0,0,0],shape=[3,3])
print(W1)
print(W2)
neigh_repre=[W1,W2]
for i in range(2):
    neigh_embs_snf=snf.SNF(neigh_repre , K=2, t=3, alpha=1.0)
    print('nes')
    print(neigh_embs_snf)


