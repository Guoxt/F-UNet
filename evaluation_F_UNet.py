#coding:utf8
import models
from config import *
import torch as t
from tqdm import tqdm
import numpy
import time
import os

check_ing_path = '/userhome/GUOXUTAO/2020_00/NET21/00/check/pthhh/00/'

check_list = os.listdir(check_ing_path)

read_list = os.listdir('/userhome/GUOXUTAO/2020_00/NET21/00/check/dicee/')

model_feature = getattr(models, 'unet_3dd')() 
model_feature.cuda()
model_feature.load_state_dict(t.load('/userhome/GUOXUTAO/2020_00/NET21/00/check/stage1_from_pth/0_4444_1_0.0001_4_0909_23:16:24.pth'))

for index,checkname in enumerate(check_list):
    
    print(index,checkname)
    
    if checkname not in read_list:
    #if 1 > 0:
    
        model = getattr(models, 'unet_3ddd')()
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
                        
                                        down_1 = model_feature(input)
                                        #print(down_1.shape)
                                        score = model(down_1)
                                        score = torch.nn.Softmax(dim=1)(score).squeeze().detach().cpu().numpy()
                                    
                                        prob[:,g+s0*i:g+s0*i+ss,g+s0*ii:g+s0*ii+ss,g+s1*iii:g+s1*iii+ss] = prob[:,g+s0*i:g+s0*i+ss,g+s0*ii:g+s0*ii+ss,g+s1*iii:g+s1*iii+ss] + score
                                        
                                        
                label = np.argmax((prob).astype(float),axis=0) 
                pre = label                                        
                 
                #print(np.sum(tru==1),np.sum(tru==2),np.sum(tru==4))
                ###################################
                ###################################
                ###################################
                ### WT 1 2 4
                ### TC 1 4
                ### ET 4
                WT_pre = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
                WT_tru = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
                TC_pre = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
                TC_tru = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
                ET_pre = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
                ET_tru = np.zeros((data.shape[1],data.shape[2],data.shape[3]))    
                
                WT_pre[pre>0] = 1
                WT_tru[tru>0] = 1
                
                TC_pre[pre==1] = 1
                TC_tru[tru==1] = 1
                TC_pre[pre==4] = 1
                TC_tru[tru==4] = 1
                
                ET_pre[pre==4] = 1
                ET_tru[tru==4] = 1
                
                a1 = np.sum(WT_pre==1)
                a2 = np.sum(WT_tru==1)
                a3 = np.sum(np.multiply(WT_pre,WT_tru)==1)
                #print(a1,a2,a3)
                if a1+a2 > 0:
                    WT_Dice = (2.0*a3)/(a1 + a2)
                WT_dice.append(WT_Dice)

                a1 = np.sum(TC_pre==1)
                a2 = np.sum(TC_tru==1)
                a3 = np.sum(np.multiply(TC_pre,TC_tru)==1)

                if a1+a2 > 0:
                    TC_Dice = (2.0*a3)/(a1 + a2)
                TC_dice.append(TC_Dice)

                a1 = np.sum(ET_pre==1)
                a2 = np.sum(ET_tru==1)
                a3 = np.sum(np.multiply(ET_pre,ET_tru)==1)

                if a1+a2 > 0:
                    ET_Dice = (2.0*a3)/(a1 + a2)
                if a1 == 0 and a2 == 0:
                    ET_Dice = 1
                ET_dice.append(ET_Dice)
                
                print(WT_Dice,TC_Dice,ET_Dice)
             
            #np.save('userhome/GUOXUTAO/2019_01/NET04/ww.npy',WT_dice)
            #np.save('userhome/GUOXUTAO/2019_01/NET04/tt.npy',TC_dice)
            #np.save('userhome/GUOXUTAO/2019_01/NET04/ee.npy',ET_dice)
            
            ### mean
            mean_WT_dice = np.mean(WT_dice)
            mean_ET_dice = np.mean(ET_dice)
            mean_TC_dice = np.mean(TC_dice)

            print('mean  ', 'WT:', mean_WT_dice,'  ', 'TC:', mean_TC_dice,'  ','ET:', mean_ET_dice)

            ### std
            std_WT_dice = np.std(WT_dice)
            std_ET_dice = np.std(ET_dice)
            std_TC_dice = np.std(TC_dice)   
            print('std  ', 'WT:', std_WT_dice,'  ', 'TC:', std_TC_dice,'  ','ET:', std_ET_dice)
        
        
            os.makedirs('/userhome/GUOXUTAO/2020_00/NET21/00/check/dicee/'+checkname+'/')
            savee = []
            savee.append(mean_WT_dice)
            savee.append(mean_ET_dice)
            savee.append(mean_TC_dice)
            np.save('/userhome/GUOXUTAO/2020_00/NET21/00/check/dicee/'+checkname+'/dice.npy',savee)
    #break