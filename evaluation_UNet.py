#coding:utf8
import models
from config import *
import torch as t
from tqdm import tqdm
import numpy
import time
import os

check_ing_path = '/userhome/GUOXUTAO/2020_00/NET21/00/check/stage1_from_pth/'

check_list = os.listdir(check_ing_path)

read_list = os.listdir('/userhome/GUOXUTAO/2020_00/NET21/00/check/dice/')

for index,checkname in enumerate(check_list):
    
    print(index,checkname)
    
    if checkname not in read_list:
    #if 1 > 0:
    
        model = getattr(models, 'unet_3d')()
        model.eval()
        model.load_state_dict(t.load(check_ing_path+checkname))
        model.eval()

        if opt.use_gpu: model.cuda()
        
        if 1 > 0:

            testpath = '/userhome/GUOXUTAO/data/datafristpaper/data/test/data/'
            folderlist = os.listdir(testpath)
            
            WT_dice = []
            TC_dice = []
            ET_dice = []            
            
            for index,fodername in enumerate(folderlist):
                print(index,fodername)
                data = np.load(testpath+fodername)
                vector = data[0:4,:,:,:]
                tru = data[4,:,:,:]
                
                prob = np.zeros((5,data.shape[1],data.shape[2],data.shape[3]))
                
                g = 10
                s0 = 32
                s1 = 32
                ss = 128
                for i in range(50):
                    for ii in range(50):
                        for iii in range(50):
                            if g+s0*i+ss < data.shape[1]-g:
                                if g+s0*ii+ss < data.shape[2]-g:
                                    if g+s1*iii+ss < data.shape[3]-g:
                                        img_out = vector[:,g+s0*i:g+s0*i+ss,g+s0*ii:g+s0*ii+ss,g+s1*iii:g+s1*iii+ss]
                                        img = torch.from_numpy(img_out).unsqueeze(0).float()
                                        with torch.no_grad():
                                            input = t.autograd.Variable(img)
                                        if True: input = input.cuda()
                        
                                        #down_1 = model_feature(input)
                                        #print(down_1.shape)
                                        score = model(input)
                                        score = torch.nn.Softmax(dim=1)(score).squeeze().detach().cpu().numpy()
                                    
                                        prob[:,g+s0*i:g+s0*i+ss,g+s0*ii:g+s0*ii+ss,g+s1*iii:g+s1*iii+ss] = prob[:,g+s0*i:g+s0*i+ss,g+s0*ii:g+s0*ii+ss,g+s1*iii:g+s1*iii+ss] + score
                                        
                                        
                label = np.argmax((prob).astype(float),axis=0) 
                pre = label                                        
                 
