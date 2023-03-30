# -*- coding: utf-8 -*-

#This module counts support vectors and generates files
#for visualization with SV_viz.py.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)


path = [".../data/ozone/ozone_",
     ".../data/scene/scene_",
     ".../data/coil/coil_",
     ".../data/thy/thy_",
     ".../data/us/us_"]

sv_path = [".../data/ozone/SV/ozone_",
     ".../data/scene/SV/scene_",
     ".../data/coil/SV/coil_",
     ".../data/thy/SV/thy_",
     ".../data/us/SV/us_"]

#base and data augmentation methods
method = ['','CS_','ROS_','SM_','AD_','rem_']
meth_eng = ['base','CS','ROS','SM','AD','rem']
m_eng_no_bs = ['CS','ROS','SM','AD','rem']

#hyper-parameters
num_meth = len(method)
num_cls=2
runs = 5
num_ds = int(5)
dims = [72,294,85,52,100]

ds_eng = ['oz','sc','co','th','us']

#class to collect information
class Dataset:
  def __init__(self, num_meth):
      
      self.syn = np.zeros(num_meth-1)
      self.new = np.zeros(num_meth-1)
      self.dc_v = np.zeros(num_meth)
      self.dc_n = np.zeros(num_meth)
      
oz = Dataset(num_meth)
sc = Dataset(num_meth)
co = Dataset(num_meth)
th = Dataset(num_meth)
us = Dataset(num_meth)

dsets = [oz,sc,co,th,us]

for d in range(num_ds): 
    print('dataset',ds_eng[d])
    
    syn_per = np.zeros((runs,num_meth-1))
    new_per = np.zeros((runs,num_meth-1))
    
    base_len = np.zeros(runs)
    dc_maj_ct = np.zeros((runs,num_meth))
    dc_min_ct = np.zeros((runs,num_meth))

    dc_maj_v1 = np.zeros((runs,num_meth))
    dc_min_v1 = np.zeros((runs,num_meth))
    
    for m in range(len(method)): 
        print('method',m,meth_eng[m])
        
        for r in range(runs):
            print('run',r)
            
            if m == 0:
                f = path[d] + 'ytrn_' + str(r) + '.csv'
                pdf = pd.read_csv(f)
                y = pdf.to_numpy()
                y = np.squeeze(y)
                base_len[r] = len(y)
            
            f = sv_path[d] + method[m] + 'dc_' + str(r) + '.csv'
            pdf = pd.read_csv(f)
            dc = pdf.to_numpy()
            dc = np.squeeze(dc)
            
            dc_min = 0
            dc_maj = 0
            
            dc_min_v = 0
            dc_maj_v = 0
            
            for n in range(len(dc)):
                if dc[n]> 0:
                    dc_min+=1
                    dc_min_v+=dc[n]
                elif dc[n]<0:
                    dc_maj+=1
                    dc_maj_v+=dc[n]
            
            dc_maj_ct[r,m]=dc_maj
            dc_min_ct[r,m]=dc_min
            
            dc_maj_v1[r,m]=dc_maj_v
            dc_min_v1[r,m]=dc_min_v
            
            f = sv_path[d] + method[m] + 'sv_ind_' + str(r) + '.csv'
            pdf = pd.read_csv(f)
            inds = pdf.to_numpy()
            inds = np.squeeze(inds)
            
            if m == 0:
                b_inds = inds
                
            else:
                m_inds = inds
                m_sv_len = len(m_inds)
                
                inter = np.intersect1d(b_inds,m_inds)
                len_inter = len(inter)
            
                new = m_sv_len - len_inter
                new = new / m_sv_len
                
                new_per[r,m-1]= new
                
                syn = 0
                for n in range(m_sv_len):
                    if m_inds[n] > base_len[r]:
                        syn+=1
                
                syn = syn / m_sv_len
                syn_per[r,m-1]= syn
    
    syn_per = np.mean(syn_per,axis=0)
    new_per = np.mean(new_per,axis=0)
    syn_per = syn_per.reshape(1,-1)
    new_per = new_per.reshape(1,-1)
    
    dc_n_rat = dc_maj_ct / dc_min_ct 
    dc_n_rat = np.mean(dc_n_rat,axis=0)
    dc_n_rat = dc_n_rat.reshape(1,-1)
    
    dc_v_rat = dc_maj_v1 / dc_min_v1 * -1
    dc_v_rat = np.mean(dc_v_rat,axis=0)   
    dc_v_rat = dc_v_rat.reshape(1,-1)
    
    meth_eng1 = ['Base','CS','ROS','SMOTE','ADASYN','REMIX']
    m_eng_no_bs1 = ['CS','ROS','SMOTE','ADASYN','REMIX']
    
    pdn = pd.DataFrame(data=dc_n_rat,columns=meth_eng1)
    f=".../figs/SV_dcn_" + \
        ds_eng[d] + '.csv'
    pdn.to_csv(f,index=False)

    pdv = pd.DataFrame(data=dc_v_rat,columns=meth_eng1)
    f=".../figs/SV_dcv_" + \
        ds_eng[d] + '.csv'
    pdv.to_csv(f,index=False)

    pds = pd.DataFrame(data=syn_per,columns=m_eng_no_bs1)
    f=".../figs/SV_syn_" + \
        ds_eng[d] + '.csv'
    pds.to_csv(f,index=False)
    
    pdnw = pd.DataFrame(data=new_per,columns=m_eng_no_bs1)
    f=".../figs/SV_new_" + \
        ds_eng[d] + '.csv'
    pdnw.to_csv(f,index=False)
    
   
