# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
import resnet_XAI as models
import torch.utils.data

#DA methods
meth_eng = ['imb', 'ros', 'rem', 'dsm', 'eos']


paths=[

#base
".../cifar_10/no_augcifar10_mod_133_0.pth", 
".../cifar_10/no_augcifar10_mod_186_10.pth",
".../cifar_10/no_augcifar10_mod_186_100.pth",

#ROS
".../cifar_10/no_augcifar10_mod_120_0.pth",
".../cifar_10/no_augcifar10_mod_122_10.pth",
".../cifar_10/ros_cifar10_mod_119_100.pth",

#REM
".../cifar_10/cifar10_rem_mod_115_0.pth",
".../cifar_10/cifar10_rem_mod_162_10.pth",
".../cifar_10/cifar10_rem_mod_146_100.pth",


#DSM
".../cifar_10/DSM_0_best.pth",
".../cifar_10/DSM_10_best.pth",
".../cifar_10/DSM_100_best.pth",

#EOS
".../cifar_10/EOS_0_best.pth",
".../cifar_10/EOS_10_best.pth",
".../cifar_10/EOS_100_best.pth",

]
runs = 3

#number of classes
num_cls = 10

#size of latent layer (FE)
num_wts = 64

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
  
        model = models.resnet32(num_classes=num_cls, use_norm=use_norm)
        
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

f='.../figs/cif_wt_diffs.csv'
pdf.to_csv(f,index=False)











"""
print('maj norms',maj_norm)
print('min norms',min_norm)

cols = ['Balanced', 'Imbalanced', 'ROS', 'REMIX', 'DSM', 'EOS' ]
maj_norm = maj_norm.reshape(1,-1)
min_norm = min_norm.reshape(1,-1)
pd1 = pd.DataFrame(data=maj_norm)
pd2 = pd.DataFrame(data=min_norm)
pdf = pd.concat([pd1,pd2],axis=0)
pdf.columns = cols
print(pdf)


f = "C:/Users/ddabl/Documents/1_diss/figs/inat_wt_norms.csv"
pdf.to_csv(f,index=False)

maj_norm = maj_norm.reshape(-1,1)
min_norm = min_norm.reshape(-1,1)
comb = np.concatenate((maj_norm,min_norm))
comb.shape

leg = np.array(['Balanced', 'Imbalanced', 'ROS', 'REMIX', 'DSM', 'EOS',
        'Balanced', 'Imbalanced', 'ROS', 'REMIX', 'DSM', 'EOS' ])

leg = leg.reshape(-1,1)

a = np.array(['Majority'] * 6)
b = np.array(['Minority'] * 6)
cols4 = np.concatenate((a,b))

#leg = np.array(['Balanced', 'Balanced','Imbalanced', 'Imbalanced',
#         'ROS', 'ROS','REMIX', 'REMIX','DSM', 'DSM','EOS', 'EOS'])
#leg = leg.reshape(-1,1)

#cols4 = np.array(['Majority','Minority'] * 6)
cols4
cols4 = cols4.reshape(-1,1)


pd1 = pd.DataFrame(data=comb,columns=['Data'])
pd2 = pd.DataFrame(data=leg,columns=['Methods'])
pd3 = pd.DataFrame(data=cols4,columns=['Legend'])

pdf = pd.concat([pd1,pd2,pd3],axis=1)
pdf.shape
pdf

f = "C:/Users/ddabl/Documents/1_diss/figs/inat_wt_norms_for_viz.csv"
pdf.to_csv(f,index=False)

for q in range(len(pdf)):
        pdf.iloc[q,0]= np.round(pdf.iloc[q,0],decimals=2)




#create grouped bar chart
ax = sns.barplot(x='Methods', y='Data',  
                 hue='Legend', data=pdf)

for ii in ax.containers:
        ax.bar_label(ii,)

plt.xlabel('')
plt.ylabel('Weight Norms')

#plt.title('LG Weight Norms')
plt.title('INaturalist Weight Norms')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
           borderaxespad=0)


f1="C:/Users/ddabl/Documents/1_diss/figs/inat_wt_norms.pdf"

plt.savefig(f1,bbox_inches='tight')
"""















  