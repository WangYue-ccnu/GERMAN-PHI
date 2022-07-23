# caculate
import numpy as np
import pandas as pd
import math


def Gussian_similarity(intMat):
    # np.savetxt('vh_admat_dgc_list.csv', result, delimiter = ',')

    gamall = 1

    # nd=max(result[:,0])

    # nl=max(result[:,1])
    nl = np.shape(intMat)[0]  # 0是横轴1是纵轴
    # nl = np.shape(intMat)[1]


    #    pp=np.shape(result)[0]
    #    qq=np.shape(result)[1]

    # calculate gamal for Gaussian kernel calculation
    sl = np.zeros(nl)

    for i in range(nl):
        # sl[i] = np.square(np.linalg.norm(intMat[:,i]))
        sl[i] = np.square(np.linalg.norm(intMat[i, :]))
    gamal = nl / sum(np.transpose(sl)) * gamall


    # hostMat = np.zeros([nl,nl],float)
    # for i in range(nl):
    #     for j in range(nl):
    #         hostMat[i, j] = math.exp(-gamal*np.square(np.linalg.norm(intMat[:, i]-intMat[:, j])))
    phageMat = np.zeros([nl, nl], float)
    for i in range(nl):
        for j in range(nl):
            phageMat[i, j] = math.exp(-gamal * np.square(np.linalg.norm(intMat[i, :] - intMat[j, :])))

    return phageMat
    # return hostMat


if __name__ == "__main__":
  # intMat = np.loadtxt("../datasets/data(71)/ph82_admat_dgc.txt") #输入0 1关联矩阵
  # hostMat=Gussian_similarity(intMat)
  p_h_true=pd.read_csv('../data/p_h_df.csv', index_col=0) #将第一列作为索引列
  intMat=p_h_true.values.astype(np.int64)
  phageMat=Gussian_similarity(intMat)
  df_p = pd.DataFrame(data=phageMat, index=p_h_true.index, columns=p_h_true.index)
  df_p.to_csv('../data/gip1330_v_df.csv')
  intMat = np.matrix(intMat).T
  hostMat = Gussian_similarity(intMat)
  df_h = pd.DataFrame(data=hostMat, index=p_h_true.columns, columns=p_h_true.columns)
  df_h.to_csv('../data/gip171_h_df.csv')
  np.savetxt('../data/giph_h_values.csv', df_h.values, delimiter=',',
             fmt='%.18f')
  np.savetxt('../data/gipv_v_values.csv', df_p.values, delimiter=',', fmt='%.18f')
  # hostMat=Gussian_similarity(intMat)[1]
#
#
# df_p=pd.DataFrame(data=phageMat,index=p_h_true.index,columns=p_h_true.index)
# df_p.to_csv('../数据/gip1330-v.csv')
# intMat=np.matrix(intMat).T
# hostMat=Gussian_similarity(intMat)
# df_h=pd.DataFrame(data=hostMat,index=p_h_true.columns,columns=p_h_true.columns)
# df_h.to_csv('../数据/gip171-h.csv')
# np.savetxt('../datasets/data(71)/gip82-p.txt', phageMat)
# np.savetxt('../datasets/data(229)/h229_admat_dg.csv', hostMat, delimiter = ',')

# np.savetxt('../datasets/data(71)/ph82_admat_dg.txt', phageMat, delimiter = ',')
# np.savetxt('../datasets/data(71)/ph82_admat_dg.txt', phageMat, delimiter = ',')
#   np.savetxt('../datasets/data(229)/gip229-p.txt', hostMat)