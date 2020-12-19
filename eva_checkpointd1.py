from __future__ import print_function, division
from torch.autograd import Variable
from spectral import SpectralNorm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pickle
import torch.nn.functional as F

from pathlib import Path
import shutil

import multiprocessing as mp
import threading as td

import numpy as np


import torch.autograd as autograd

from senv3 import ActiveVisionDatasetEnv
from torch.autograd import Variable as V
import futild as futil
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Navigation(nn.Module):
    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Navigation, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(2, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, curr_dim, 4))
        self.last = nn.Sequential(*last)#512
        #===========================================================================================
        self.f0=nn.Linear(512,512)
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,512)

        self.fc3=nn.Linear(512*4,1024)
        self.fc4=nn.Linear(1024,512)
        self.fc_mean=nn.Linear(512,512)
        self.fc_sigma=nn.Linear(512,512)
        self.fz1=nn.Linear(512,512)
        self.fz2=nn.Linear(512,512)
        #*******************
        self.fca=nn.Linear(2,512)
        self.fpa=nn.Linear(7,512)
        self.fc50=nn.Linear(1024+512,1024)
        #********************
        self.fc5=nn.Linear(1024,512)
        self.fc6=nn.Linear(512,256)
        self.actor=nn.Linear(256,7)
        #==============================================================================================
        self.fp1=nn.Linear(7,512)
        self.fp2=nn.Linear(1024,512)
        self.fp3=nn.Linear(512,512)
        self.fp_mean=nn.Linear(512,512)
        self.fp_sigma=nn.Linear(512,512)
        #===========goal checker==========================
        self.gc1=nn.Linear(2048,1024)
        self.gc2=nn.Linear(1024,512)
        self.gc3=nn.Linear(512,2)
        #===============collision checker======
        self.cc1=nn.Linear(2048,1024)
        self.cc2=nn.Linear(1024,512)
        self.cc3=nn.Linear(512,7)
        self.sigmoid=nn.Sigmoid()


    def forward_once(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out=self.l4(out)
        out=self.last(out)
        out = out.view(out.size(0), -1)
        out=F.relu(self.f0(out))
        return out

    def forward_prior(self,x,a):
        x=self.forward_once(x)        
        softplus = nn.Softplus()
        pa=F.relu(self.fp1(a))
        px=torch.cat((x,pa),1)
        px=F.relu(self.fp2(px))
        px=F.relu(self.fp3(px))
        p_mean=self.fp_mean(px)
        p_sigma=softplus(self.fp_sigma(px))
        self.batch_size=p_mean.size()[0]
        pz=p_mean+torch.exp(p_sigma/2)*torch.randn(self.batch_size, 512, device=torch.device("cuda")) 
        pz=F.relu(self.fz1(pz))
        pz=F.relu(self.fz2(pz))
        return p_mean,p_sigma, pz

    def forward(self, px1,px2,px3, x,g,pre_action):
        self.batch_size=x.size()[0]
        px1 = self.forward_once(px1)
        px2 = self.forward_once(px2)
        px3 = self.forward_once(px3)
        c_x = self.forward_once(x)
        #============================================================
        cc=torch.cat((px1,px2,px3,c_x),1)
        cc=F.relu(self.cc1(cc))
        cc=F.relu(self.cc2(cc))#512
        cc=self.cc3(cc)#7
        #============================================================
        x=c_x
        g =self.forward_once(g)
        px1=torch.cat((px1,g),1)
        px1=F.relu(self.fc1(px1))
        px1=F.relu(self.fc2(px1))
        px2=torch.cat((px2,g),1)
        px2=F.relu(self.fc1(px2))
        px2=F.relu(self.fc2(px2))
        px3=torch.cat((px3,g),1)
        px3=F.relu(self.fc1(px3))
        px3=F.relu(self.fc2(px3))
        x=torch.cat((x,g),1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        xx=torch.cat((px1,px2,px3,x),1)
        #==========================================
        x=F.relu(self.fc3(xx))
        x=F.relu(self.fc4(x))#512
        softplus = nn.Softplus()
        z_mean=self.fc_mean(x)
        z_sigma=softplus(self.fc_sigma(x))
        z=z_mean+torch.exp(z_sigma/2)*torch.randn(self.batch_size, 512, device=torch.device("cuda"))    
        z=F.relu(self.fz1(z))
        z=F.relu(self.fz2(z))
        pre_a=F.relu(self.fpa(pre_action[:,:7])) 
        # c_a=F.relu(self.fca(pre_action[:,7:])) 
        y=torch.cat((c_x,z,pre_a),1)
        y=F.relu(self.fc50(y))
        y=F.relu(self.fc5(y))
        y=F.relu(self.fc6(y))
        logit = self.actor(y)
        #=================================
        gc=F.relu(self.gc1(xx))
        gc=F.relu(self.gc2(gc))#512
        gc=self.sigmoid(self.gc3(gc))#2    
        #==========================
        return logit, gc, cc, z_mean,z_sigma, z 


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)


    def chooseact(self, px1,px2,px3, x, g, pre_action):
        self.batch_size=x.size()[0]
        c_x=x
        px1=torch.cat((px1,g),1)
        px1=F.relu(self.fc1(px1))
        px1=F.relu(self.fc2(px1))
        px2=torch.cat((px2,g),1)
        px2=F.relu(self.fc1(px2))
        px2=F.relu(self.fc2(px2))
        px3=torch.cat((px3,g),1)
        px3=F.relu(self.fc1(px3))
        px3=F.relu(self.fc2(px3))
        x=torch.cat((x,g),1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        xx=torch.cat((px1,px2,px3,x),1)
        x=F.relu(self.fc3(xx))
        x=F.relu(self.fc4(x))#512
        softplus = nn.Softplus()
        z_mean=self.fc_mean(x)
        z_sigma=softplus(self.fc_sigma(x))
        z=z_mean+torch.exp(z_sigma/2)*torch.randn(self.batch_size, 512, device=torch.device("cuda"))    
        z=F.relu(self.fz1(z))
        z=F.relu(self.fz2(z))
        pre_a=F.relu(self.fpa(pre_action[:,:7])) 
        y=torch.cat((c_x,z,pre_a),1)
        y=F.relu(self.fc50(y))
        y=F.relu(self.fc5(y))
        y=F.relu(self.fc6(y))
        #==========================
        gc=F.relu(self.gc1(xx))
        gc=F.relu(self.gc2(gc))#512
        gc=self.sigmoid(self.gc3(gc))#2
        logit = self.actor(y)
        probs = F.softmax(logit,dim=1)
        action = probs.multinomial(1, replacement=False).view(-1).data
        #==========================
        _,m_or_s =torch.max(gc,1)
        a=torch.tensor(1, device='cuda:0')
        b=torch.tensor(6, device='cuda:0')
        for i in range(self.batch_size):#num_envs
            if action[i]==b:
                x=logit[i][:-1]
                action[i]=x.argmax()
            if m_or_s[i]==a:
                action[i]=b
        return action
       
#=====================================================================================================
num_actions = 7

actor_critic =Navigation()
USE_CUDA=True
if USE_CUDA:
    actor_critic  = actor_critic.cuda()

CHECK_DIR="./checkpointy/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#===============================================================================================
testpath=[]
with open("./testdata/Onediff",'rb') as fp:
    testpath1=pickle.load(fp)
    testpath1=testpath1[:500]
testpath=testpath1
with open("./testdata/Twodiff",'rb') as fp:
    testpath2=pickle.load(fp)
    testpath2=testpath2[:500]
testpath.extend(testpath2)
with open("./testdata/Threediff",'rb') as fp:
    testpath3=pickle.load(fp)
    testpath3=testpath3[:500]
testpath.extend(testpath3)
with open("./testdata/Fourdiff",'rb') as fp:
    testpath4=pickle.load(fp)
    testpath4=testpath4[:500]
testpath.extend(testpath4)
with open("./testdata/Fivediff",'rb') as fp:
    testpath5=pickle.load(fp)
    testpath5=testpath5[:500]
testpath.extend(testpath5)


print("test path:",len(testpath))
#==================================================================================================
SR=[]
SPL=[]
CSR=[]
CSPL=[]

fele='best.ckpt'
best_path = os.path.join(CHECK_DIR, fele) #checkpoint-1186703
if os.path.isfile(best_path):      
    checkpoint = torch.load(best_path,map_location=device)
    global_t=checkpoint['global_t']
    actor_critic.load_state_dict(checkpoint['state_dict'],strict=False)#====================================
    print("=> loaded checkpoint '{}' (global_t {})"
        .format(best_path, checkpoint['global_t']))
else:
    global_t=0
    print("=> no checkpoint found at '{}'".format(best_path))
#===================================================
successtime=0
spl=0
with torch.set_grad_enabled(False):
#============extract feature===========================
    featrain_iter = futil.create_iterator()
    featrain_loader = torch.utils.data.DataLoader(
            featrain_iter,
            batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    feades={}
    for inputs, labels in featrain_loader:          
        inputs = inputs.to(device)
        fdes=actor_critic.forward_once(inputs)#2048***************************
        for i in range(fdes.shape[0]):
            feades[labels[i]]=fdes[i]
    print(len(feades))
    with open('featuredes', 'wb') as fp:
        pickle.dump(feades, fp)
    tenv = ActiveVisionDatasetEnv()
#=====================================================================================================================================
    count=0
    category_sr=[0,0,0,0,0]
    category_spl=[0,0,0,0,0]
    for ele in testpath:
        world=ele[0]
        startid=ele[1]
        endid=ele[2]
        step=0
        current_pim1,current_pim2,current_pim3,current_state,current_g,pre_action = tenv.reset(world,startid,endid)
        while step<100:
            action = actor_critic.chooseact(current_pim1.unsqueeze(0),current_pim2.unsqueeze(0),current_pim3.unsqueeze(0),current_state.unsqueeze(0),current_g.unsqueeze(0),pre_action.unsqueeze(0))
            current_pim1,current_pim2,current_pim3,current_state,current_g, reward, done,gt_action, shortest,pre_action = tenv.step(action.cpu().data.numpy()[0])
            step+=1 
            if done and tenv.reward>5.0:
                current_vertex=tenv.current_vertex
                #===============================================
                vex=tenv.id_to_index[startid]
                slen=len(tenv.shortest_path(vex,current_vertex))
                mylen=step
                spl+=slen/max(mylen,slen)
                #================================================
                successtime+=1
                # if slen>15:
                #     endid=tenv.index_to_id[current_vertex]
                #     with open('mypath/'+world+'_'+ele[1]+'_'+endid+'.txt', 'wb') as fp:
                #         pickle.dump(tenv.mypath, fp)   
                break
            elif done:
                break
        count+=1
        if count%500==0:#500
            idx=int(count/500)-1#500
            category_sr[idx]=successtime
            category_spl[idx]=spl
    spl=spl/2500#500
    successtime=successtime/2500#500
    SPL.append(spl)
    SR.append(successtime)   
    print("SPL:",SPL)
    print("SR:",SR) 

    for idx in range(1,5):#[1,2,3,4]
        category_spl[5-idx]=category_spl[5-idx]-category_spl[4-idx]
        category_sr[5-idx]=category_sr[5-idx]-category_sr[4-idx]
    category_spl=np.array(category_spl)/500#500
    category_sr=np.array(category_sr)/500#500
    CSR.append(category_sr)
    CSPL.append(category_spl)

    print("CSPL:",CSPL)
    print("CSR:",CSR) 








    
    





