#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:15:07 2018

@author: yanxinzhou
"""

import numpy as np
import pandas as pd
import scipy
#import seaborn as sns
from numpy.linalg import inv
from scipy.stats import invgamma
from numpy import linalg as La
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from scipy.stats import multivariate_normal
import pdb as db

mu, sigma = 0, 1 # mean and standard deviation
m,n=5,60
data=np.zeros((n,m))

for i in range(m):
    s = np.random.normal(mu, sigma, n)
    data[:,i]=s

df=pd.DataFrame(data,columns={'x1','x2','x3','x4','x5'})

mu, sigma = 0, 2.5
eps = np.random.normal(mu, sigma, n)
target=df['x4']+1.2*df['x5']+eps

maxiters=5000
data=df.values
y=target.values
pi,c,lambda_gamma,v=1/2, 10 ,1,0#data.shape[0]
R=df.cov().values
#R=np.zeros((5,5))
#np.fill_diagonal(R,1)

beta=np.zeros((maxiters,m))
sigma=np.zeros((maxiters,1))
r=np.zeros((maxiters,m))

r[0]=np.ones(m)
a=inv(np.matmul(data.T,data))
b=np.matmul(data.T,y)
beta[0]=np.matmul(a,b)

sigma[0]=np.sqrt((y-(beta[0]*data).sum(1)).var())
#db.set_trace()

a=np.zeros((maxiters,m))
a[np.where(r==0)[0],np.where(r==0)[1]]=1
a[np.where(r==1)[0],np.where(r==1)[1]]=c

ssxx=((data-data.mean(0))**2).sum(0)
tau=sigma/np.sqrt(ssxx)

temp=a*tau
D=[]
for i in range(temp.shape[0]):
    D.append(np.diag(temp[0]))
D=np.array(D)

def get_A(sigma_1,x,D_1,R):
    try:
        A = sigma_1**(-2)*np.matmul(x.T,x)+np.matmul(np.matmul(inv(D_1).T,inv(R)),inv(D_1))##!!!!get rid of transpose
    except:
        db.set_trace()
    return inv(A)
      
def get_beta(sigma_1,x,D_1,R,beta_ls):
    
    beta_ls=beta_ls.reshape((beta_ls.shape[0],1))
    A=get_A(sigma_1,x,D_1,R)
    
    xx=np.matmul(x.T,x)
    temp=np.matmul(A,xx)  #get ride of transpose!!!!!
    temp=np.matmul(temp,beta_ls)# 5*1
    
    mean=(sigma_1**(-2)*temp).reshape(temp.shape[0])
    cov=A#5*5
    #db.set_trace()
    beta=np.random.multivariate_normal(mean, cov, 1)[0]
    #db.set_trace()
    return beta
    
def get_sigma(n,y,beta,r,lambda_gamma,v):
    

    err=((y-(beta*data).sum(1))**2).sum()
    a=(n+v)/2
    scale=(err+v*lambda_gamma)/2
    sig=invgamma.rvs(a=a,loc=0,scale=scale,size=1)
    #db.set_trace()
    return sig

def get_gamma(idx,beta):
    
    r[idx]=r[idx-1]
    sig=np.sqrt((y-(beta*data).sum(1)).var())
    tau[idx]=sig/np.sqrt(ssxx)
    
    a1=np.zeros((m))
    a1[r[idx]==0]=1
    a1[r[idx]==1]=c
    d=np.diag(a1*tau)
    for i in range(0,len(beta)):
        
        a1[i]=c
        d1=np.diag(a1*tau[idx])
        mean1,sigma1=0,np.matmul(np.matmul(d1.T,R),d1)
        aa=multivariate_normal.pdf(beta, mean=np.zeros((5)), cov=sigma1)
        #aa=(1/np.sqrt(La.norm(sigma1)))*np.exp(-0.5*np.matmul(np.matmul(beta,inv(sigma1)),beta))
        
        a2=a1.copy()
        a2[i]=1
        d2=np.diag(a2*tau[idx])
        mean2,sigma2=0,np.matmul(np.matmul(d2.T,R),d2) 
        bb=multivariate_normal.pdf(beta, mean=np.zeros((5)), cov=sigma2)
#        bb=(1/np.sqrt(La.norm(sigma2)))*np.exp(-0.5*np.matmul(np.matmul(beta,inv(sigma2)),beta))
        
        if (aa+bb)!=0:  
            p=aa/(aa+bb)
        else:
            db.set_trace()
        
        #db.set_trace()
        
        if p<0.5:
            r[idx,i]=1
        else:
            r[idx,i]=0

#    a1[r[idx]==0]=1
#    a1[r[idx]==1]=c
#    D[idx]=np.diag(a1*tau[idx])
    #db.set_trace()
    return r[idx]
      
    
for i in range(1,len(r)):#len(r)):
    #db.set_trace()
    beta[i]=get_beta(sigma[i-1],data,D[i-1],R,beta[0])
    sigma[i]= get_sigma(n,y,beta[i],r[i-1],lambda_gamma,v)
    r[i]=get_gamma(i,beta[i])
    #db.set_trace()
    
unique_elements, counts_elements = np.unique(r,axis=0, return_counts=True)
rank=list(zip(unique_elements, counts_elements))
rank=rank[:2500]
rank=sorted(rank, key=lambda rank: rank[1],reverse=True) 
rank