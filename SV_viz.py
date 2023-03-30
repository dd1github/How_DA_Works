# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# This file generates the following visualizations:
#    Ratio of Class Dual Coefficient Values, 
#    Ratio of Number of Class Support Vectors, 
#    Ratio of New Support Vectors vs Base, and 
#    Ratio of Synthetic Support Vectors.

#In order to create these visualizations, insert the file paths below.


#dataset names
ds_eng1 = ['Ozone','Scene','Coil','Thyroid','US']
ds_eng = ['oz','sc','co','th','us']

#data augmentation and base imbalanced methods
meth_eng = ['Base','CS','ROS','SMOTE','ADASYN','REMIX']
m_eng_no_bs = ['CS','ROS','SMOTE','ADASYN','REMIX']

###############################
#Ratio of Class Dual Coefficient Values visualization

#lists to collect output
dat = []
leg = []
ds = []

for d in range(len(ds_eng)):
    file =".../figs/SV_dcv_" + \
        ds_eng[d] + '.csv'
    pdf = pd.read_csv(file)
    x = pdf.to_numpy()
    x = np.squeeze(x)
    
    
    for i in range(len(x)):
        leg.append(meth_eng[i])
        dat.append(x[i])
        ds.append(ds_eng1[d])

#convert lists to numpy arrays
dat = np.array(dat).reshape(-1,1)
leg  = np.array(leg).reshape(-1,1)
ds = np.array(ds).reshape(-1,1)

#prepare pandas files for visualization
pd1 = pd.DataFrame(data=dat,columns=['Data'])
pd2 = pd.DataFrame(data=leg,columns=['Legend'])
pd3 = pd.DataFrame(data=ds,columns=['Dataset'])

comb = pd.concat([pd1,pd2,pd3],axis=1)

ax = sns.barplot(x='Dataset', y='Data',  
             hue='Legend', data=comb,ci=None) 

plt.xlabel('Datasets')
plt.ylabel('Ratio')

plt.title('Ratio of Class Dual Coefficient Values')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
   borderaxespad=0)

f1='.../figs/SV_dcv.pdf'

plt.savefig(f1,bbox_inches='tight')

plt.show()

#################################################################

ds_eng1 = ['Ozone','Scene','Coil','Thyroid','US']
ds_eng = ['oz','sc','co','th','us']

meth_eng = ['Base','CS','ROS','SMOTE','ADASYN','REMIX']
m_eng_no_bs = ['CS','ROS','SMOTE','ADASYN','REMIX']

################################################################
#Ratio of Number of Class Support Vectors visualization

#lists to collect output
dat = []
leg = []
ds = []

for d in range(len(ds_eng)):
    f=".../figs/SV_dcn_" + \
        ds_eng[d] + '.csv'
    pdf = pd.read_csv(f)
    x = pdf.to_numpy()
    x = np.squeeze(x)
    
    
    for i in range(len(x)):
        leg.append(meth_eng[i])
        dat.append(x[i])
        ds.append(ds_eng1[d])
        
dat = np.array(dat).reshape(-1,1)
leg  = np.array(leg).reshape(-1,1)
ds = np.array(ds).reshape(-1,1)


pd1 = pd.DataFrame(data=dat,columns=['Data'])
pd2 = pd.DataFrame(data=leg,columns=['Legend'])
pd3 = pd.DataFrame(data=ds,columns=['Dataset'])

comb = pd.concat([pd1,pd2,pd3],axis=1)

ax = sns.barplot(x='Dataset', y='Data',  
             hue='Legend', data=comb,ci=None) 

plt.xlabel('Datasets')
plt.ylabel('Ratio')

plt.title('Ratio of Number of Class Support Vectors')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
   borderaxespad=0)

f1='.../figs/SV_dcn.pdf'

plt.savefig(f1,bbox_inches='tight')

plt.show()

################################################################

ds_eng1 = ['Ozone','Scene','Coil','Thyroid','US']
ds_eng = ['oz','sc','co','th','us']

meth_eng = ['Base','CS','ROS','SMOTE','ADASYN','REMIX']
m_eng_no_bs = ['CS','ROS','SMOTE','ADASYN','REMIX']

##############################################################
#Ratio of New Support Vectors vs Base viisualization

dat = []
leg = []
ds = []

for d in range(len(ds_eng)):
    f=".../figs/SV_new_" + \
        ds_eng[d] + '.csv'
    pdf = pd.read_csv(f)
    x = pdf.to_numpy()
    x = np.squeeze(x)
    
    
    for i in range(len(x)):
        leg.append(m_eng_no_bs[i])
        dat.append(x[i])
        ds.append(ds_eng1[d])
        
dat = np.array(dat).reshape(-1,1)
leg  = np.array(leg).reshape(-1,1)
ds = np.array(ds).reshape(-1,1)

pd1 = pd.DataFrame(data=dat,columns=['Data'])
pd2 = pd.DataFrame(data=leg,columns=['Legend'])
pd3 = pd.DataFrame(data=ds,columns=['Dataset'])

comb = pd.concat([pd1,pd2,pd3],axis=1)

ax = sns.barplot(x='Dataset', y='Data',  
             hue='Legend', data=comb,ci=None) 

plt.xlabel('Datasets')
plt.ylabel('Ratio')

plt.title('Ratio of New Support Vectors vs Base')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
   borderaxespad=0)

f1='.../figs/SV_new.pdf'

plt.savefig(f1,bbox_inches='tight')

plt.show()

################################################################

ds_eng1 = ['Ozone','Scene','Coil','Thyroid','US']
ds_eng = ['oz','sc','co','th','us']

meth_eng = ['Base','CS','ROS','SMOTE','ADASYN','REMIX']
m_eng_no_bs = ['CS','ROS','SMOTE','ADASYN','REMIX']

##################################################################
#Ratio of Synthetic Support Vectors visualization


dat = []
leg = []
ds = []

for d in range(len(ds_eng)):
    
    
    f=".../figs/SV_syn_" + \
        ds_eng[d] + '.csv'
    pdf = pd.read_csv(f)
    x = pdf.to_numpy()
    x = np.squeeze(x)
    
    
    for i in range(len(x)):
        if i == 2 or i == 3:
            leg.append(m_eng_no_bs[i])
            dat.append(x[i])
            ds.append(ds_eng1[d])
        
dat = np.array(dat).reshape(-1,1)
leg  = np.array(leg).reshape(-1,1)
ds = np.array(ds).reshape(-1,1)

pd1 = pd.DataFrame(data=dat,columns=['Data'])
pd2 = pd.DataFrame(data=leg,columns=['Legend'])
pd3 = pd.DataFrame(data=ds,columns=['Dataset'])

comb = pd.concat([pd1,pd2,pd3],axis=1)

ax = sns.barplot(x='Dataset', y='Data',  
             hue='Legend', data=comb,ci=None) 

plt.xlabel('Datasets')
plt.ylabel('Ratio')

plt.title('Ratio of Synthetic Support Vectors')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
   borderaxespad=0)

f1='.../figs/SV_syn.pdf'
plt.savefig(f1,bbox_inches='tight')

plt.show()






