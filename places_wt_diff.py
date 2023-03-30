# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
import resnet_XAI as models
import torch.utils.data

#dataset of methods
meth_eng = ['imb', 'ros', 'rem', 'dsm', 'eos']

paths=[

#base imbalanced
".../places/_aug_Plc_mod_25_0.pth", 
".../places/_aug_Plc_mod_17_10.pth",
".../places/_aug_Plc_mod_27_100.pth",

#ROS
".../places/_ros_aug_Plc_mod_14_0.pth",
".../places/_ros_aug_Plc_mod_26_10.pth",
".../places/_ros_aug_Plc_mod_19_100.pth",


#REM
".../places/_rem_Plc_mod_38_0.pth",
".../places/_rem_Plc_mod_24_10.pth",
".../places/_rem_Plc_mod_19_100.pth",


#DSM
".../places/plc_DSM_0_comb.pth",
".../places/plc_DSM_10_comb.pth",
".../places/plc_DSM_100_comb.pth",

#EOS
".../places/plc_EOS_0_comb.pth",
".../places/plc_EOS_10_comb.pth",
".../places/plc_EOS_100_comb.pth",
]


# number of runs
runs = 3

#number of classes
num_cls = 5

#dimension of latent space (FE)
num_wts = 64

#files to collect output
imb = np.zeros((runs*num_cls,num_wts))
ros = np.zeros((runs*num_cls,num_wts))
rem = np.zeros((runs*num_cls,num_wts))
dsm = np.zeros((runs*num_cls,num_wts))
eos = np.zeros((runs*num_cls,num_wts))

#DA methods
meths = [imb,ros,rem,dsm,eos]


model_count = 0   
for m in range(len(meth_eng)): 
    print('method',m,meth_eng[m])
    
    row_count = 0
    for r in range(runs):
        print('run',r)
        print('model',model_count)
        print(paths[model_count])

        use_norm = False  
  
        model = models.resnet56(num_classes=num_cls, use_norm=use_norm) 
        
        torch.cuda.set_device(0)
        model = model.cuda(0)

        model.load_state_dict(torch.load(paths[model_count]))
        
        wt = model.linear.weight
        wt = wt.detach().cpu().numpy()
        print(wt.shape)
        meth = meths[m]
        print(meth.shape)
        print(row_count,row_count+num_cls)
        meth[row_count:row_count+num_cls,:]= wt
        row_count+=num_cls
        model_count+=1

        
for m in range(len(meth_eng)):
    print(meth_eng[m])
    print(meths[m])
    print()

len_meth = len(meth_eng)
diffs = np.zeros(len_meth-1)

count=0
for m in range(1,len_meth):
    
    diff = np.abs(meths[m] - meths[0])
    
    diff = diff / np.abs(meths[0])
    diff = np.mean(diff)
    diffs[count]=diff
    count+=1
    
print(diffs)

dif = diffs.reshape(1,-1)
pdf = pd.DataFrame(data=dif,columns=[['ROS',
                'REMIX', 'DSM', 'EOS']])

f='.../figs/plc_wt_diffs.csv'

pdf.to_csv(f,index=False)















  