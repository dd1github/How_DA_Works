# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
import resnet_XAI as models
import torch.utils.data

#dataset of methods
meth_eng = ['imb', 'ros', 'rem', 'dsm', 'eos']


paths=[

#base
".../INAT/_aug_INat_mod_37_0.pth", 
".../INAT/_aug_INat_mod_38_10.pth",
".../INAT/_aug_INat_mod_36_100.pth",

#ROS
".../INAT/ROS_aug_INat_mod_32_0.pth",
".../INAT/ROS_aug_INat_mod_33_10.pth",
".../INAT/ROS_aug_INat_mod_36_100.pth",

#REM
".../INAT/rem_INat_mod_35_0.pth",
".../INAT/rem_INat_mod_35_10.pth",
".../INAT/rem_INat_mod_27_100.pth",

#DSM
".../INAT/inat_DSM_0_comb.pth",
".../INAT/inat_DSM_10_comb.pth",
".../INAT/inat_DSM_100_comb.pth",

#EOS
".../INAT/inat_EOS_0_comb.pth",
".../INATinat_EOS_10_comb.pth",
".../INAT/inat_EOS_100_comb.pth",
]

runs = 3
num_cls = 5
num_wts = 64

#DA methods - files to collect ouput
imb = np.zeros((runs*num_cls,num_wts))
ros = np.zeros((runs*num_cls,num_wts))
rem = np.zeros((runs*num_cls,num_wts))
dsm = np.zeros((runs*num_cls,num_wts))
eos = np.zeros((runs*num_cls,num_wts))

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

f= ".../INAT/inat_wt_diffs.csv" 
pdf.to_csv(f,index=False)
    












  