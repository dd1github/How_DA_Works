# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LG

import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

np.random.seed(seed=0)

#paths to data files
p = [".../data/ozone/ozone_",
     ".../data/scene/scene_",
     ".../data/coil/coil_",
     ".../data/thy/thy_",
     ".../data/us/us_"]


#datasets and DA methods
ds_eng = ['oz','sc','co','th','us']
method = ['','', 'ROS_','SM_','AD_','rem_']
meth_eng = ['base','CS', 'ROS_','SM_','AD_','rem_']

#hyper-parameters
num_meth = len(method)
topk = 10
num_cls=2
runs = 5
num_ds = 5
dims = [72,294,85,52,100]


#class to collect output
class Dataset:
  def __init__(self, runs, topk,dim):

    self.idx_b0 = np.zeros((runs,topk))
    self.mag_b0 = np.zeros((runs,topk))
    self.trn_mu_b = np.zeros((runs, dim))
    
    self.idx_b1 = np.zeros((runs,topk))
    self.mag_b1 = np.zeros((runs,topk))
    self.trn_std_b = np.zeros((runs, dim))
    ###################################
    self.idx_c0 = np.zeros((runs,topk))
    self.mag_c0 = np.zeros((runs,topk))
    self.trn_mu_c = np.zeros((runs, dim))
    
    self.idx_c1 = np.zeros((runs,topk))
    self.mag_c1 = np.zeros((runs,topk))
    self.trn_std_c = np.zeros((runs, dim))
    #####################################
    self.idx_r0 = np.zeros((runs,topk))
    self.mag_r0 = np.zeros((runs,topk))
    self.trn_mu_r = np.zeros((runs, dim))
  
    self.idx_r1 = np.zeros((runs,topk))
    self.mag_r1 = np.zeros((runs,topk))
    self.trn_std_r = np.zeros((runs, dim))
    ###################################
    self.idx_s0 = np.zeros((runs,topk))
    self.mag_s0 = np.zeros((runs,topk))
    self.trn_mu_s = np.zeros((runs, dim))
  
    self.idx_s1 = np.zeros((runs,topk))
    self.mag_s1 = np.zeros((runs,topk))
    self.trn_std_s = np.zeros((runs, dim))
    #####################################
    self.idx_a0 = np.zeros((runs,topk))
    self.mag_a0 = np.zeros((runs,topk))
    self.trn_mu_a = np.zeros((runs, dim))

    self.idx_a1 = np.zeros((runs,topk))
    self.mag_a1 = np.zeros((runs,topk))
    self.trn_std_a = np.zeros((runs, dim))
    ###################################
    self.idx_x0 = np.zeros((runs,topk))
    self.mag_x0 = np.zeros((runs,topk))
    self.trn_mu_x = np.zeros((runs, dim))

    self.idx_x1 = np.zeros((runs,topk))
    self.mag_x1 = np.zeros((runs,topk))
    self.trn_std_x = np.zeros((runs, dim))

#to collect individual dataset output    
oz = Dataset(runs,topk,dims[0])
sc = Dataset(runs,topk,dims[1])
co = Dataset(runs,topk,dims[2])
thy = Dataset(runs,topk,dims[3])
us = Dataset(runs,topk,dims[4])

ds_sets = [oz,sc,co,thy,us]


for d in range(num_ds): 
    print('dataset',ds_eng[d])
    dss = ds_sets[d]
    p1 = p[d]
    
    for m in range(len(method)): 
        
        print('method',m,meth_eng[m])
        
        for i in range(runs):
            print('run',i)
            
            f = p1 + 'xtrn_' + method[m] + str(i) + '.csv'
            pdf = pd.read_csv(f)
            x = pdf.to_numpy()
            
            n_cols = x.shape[1]
            
            f = p1 + 'ytrn_' + method[m] + str(i) + '.csv'
            pdf = pd.read_csv(f)
            y = pdf.to_numpy()
            y = np.squeeze(y)
            
            f = p1 + 'xtst_' + str(i) + '.csv'
            pdf = pd.read_csv(f)
            xtst = pdf.to_numpy()
            
            f = p1 + 'ytst_' + str(i) + '.csv'
            pdf = pd.read_csv(f)
            ytst = pdf.to_numpy()
            ytst = np.squeeze(ytst)
            
            if m == 0:
                dss.trn_mu_b[i,:]= np.mean(x,axis=0)
                dss.trn_std_b[i,:] = np.std(x,axis=0)
            if m == 1:
                dss.trn_mu_c[i,:]= np.mean(x,axis=0)
                dss.trn_std_c[i,:] = np.std(x,axis=0)
            if m == 2:
                dss.trn_mu_r[i,:]= np.mean(x,axis=0)
                dss.trn_std_r[i,:] = np.std(x,axis=0)
            if m == 3:
                dss.trn_mu_s[i,:]= np.mean(x,axis=0)
                dss.trn_std_s[i,:] = np.std(x,axis=0)
            if m == 4:
                dss.trn_mu_a[i,:]= np.mean(x,axis=0)
                dss.trn_std_a[i,:] = np.std(x,axis=0)
            if m == 5:
                dss.trn_mu_x[i,:]= np.mean(x,axis=0)
                dss.trn_std_x[i,:] = np.std(x,axis=0)
            
            if m == 1:
                mod = LG(class_weight='balanced')
            else:
                mod = LG()
                
            mod.fit(x,y)
            
            y_pred = mod.predict(xtst)
            
            cf = mod.coef_
            
            CE = xtst * cf
            
            for c in range(num_cls):
                count=0
                
                CE_cls = CE[ytst==c]
                yp = y_pred[ytst==c]
                
                CE_cls = CE_cls[yp==c]
                
                CE_cls = np.abs(CE_cls)
                
                ce_mu = np.mean(CE_cls,axis=0)
                
                ce_mu_s = np.sort(ce_mu)
                ce_mu_a = np.argsort(ce_mu)
                
                ce_mu_sort = ce_mu_s[::-1]
                ce_mu_arg = ce_mu_a[::-1]
                
                if c == 0:
                    
                    if m == 0:
                        dss.idx_b0[i,:]= ce_mu_arg[:10]
                        dss.mag_b0[i,:] = ce_mu_sort[:10]
                    if m == 1:
                        dss.idx_c0[i,:]= ce_mu_arg[:10]
                        dss.mag_c0[i,:] = ce_mu_sort[:10]
                    if m == 2:
                        dss.idx_r0[i,:]= ce_mu_arg[:10]
                        dss.mag_r0[i,:] = ce_mu_sort[:10]
                    if m == 3:
                        dss.idx_s0[i,:]= ce_mu_arg[:10]
                        dss.mag_s0[i,:] = ce_mu_sort[:10]
                    if m == 4:
                        dss.idx_a0[i,:]= ce_mu_arg[:10]
                        dss.mag_a0[i,:] = ce_mu_sort[:10]
                    if m == 5:
                        dss.idx_x0[i,:]= ce_mu_arg[:10]
                        dss.mag_x0[i,:] = ce_mu_sort[:10]
                    
                else:
                    
                    if m == 0:
                        dss.idx_b1[i,:]= ce_mu_arg[:10]
                        dss.mag_b1[i,:] = ce_mu_sort[:10]
                    if m == 1:
                        dss.idx_c1[i,:]= ce_mu_arg[:10]
                        dss.mag_c1[i,:] = ce_mu_sort[:10]
                    if m == 2:
                        dss.idx_r1[i,:]= ce_mu_arg[:10]
                        dss.mag_r1[i,:] = ce_mu_sort[:10]
                    if m == 3:
                        dss.idx_s1[i,:]= ce_mu_arg[:10]
                        dss.mag_s1[i,:] = ce_mu_sort[:10]
                    if m == 4:
                        dss.idx_a1[i,:]= ce_mu_arg[:10]
                        dss.mag_a1[i,:] = ce_mu_sort[:10]
                    if m == 5:
                        dss.idx_x1[i,:]= ce_mu_arg[:10]
                        dss.mag_x1[i,:] = ce_mu_sort[:10]
        
############################################################

#overlap

ds_summ = np.zeros((num_ds,num_meth-1))

for d in range(num_ds):
    dss = ds_sets[d]
    
    dsm = np.zeros((runs,num_meth-1))
    
    for r in range(runs):
        
        for m in range(num_meth):
            
            if m == 0:
                b0 = dss.idx_b0[r,:]
                b1 = dss.idx_b1[r,:]
            elif m == 1:
                m0 = dss.idx_c0[r,:]
                m1 = dss.idx_c1[r,:]
            elif m == 2:
                m0 = dss.idx_r0[r,:]
                m1 = dss.idx_r1[r,:]   
            elif m == 3:
                m0 = dss.idx_s0[r,:]
                m1 = dss.idx_s1[r,:]    
            elif m == 4:
                m0 = dss.idx_a0[r,:]
                m1 = dss.idx_a1[r,:]   
            elif m == 5:
                m0 = dss.idx_x0[r,:]
                m1 = dss.idx_x1[r,:]        
            
            if m > 0:
                z = np.intersect1d(b0,m0)
                z1 = np.intersect1d(b1,m1)
                
                dsm[r,m-1] = (len(z) + len(z1)) / 20
    
    dsmm = np.mean(dsm,axis=0)
    ds_summ[d,:]= dsmm
    
    
print(ds_summ)

print(np.mean(ds_summ,axis=0))
print(np.mean(ds_summ))
