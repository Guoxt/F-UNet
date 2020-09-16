#coding:utf8
import models
from param import *
from data.dataset import dataread
from torch.utils.data import DataLoader
import torch as t
from tqdm import tqdm
import numpy
import time
import os
import random

############################################################################
def val(model,model_feature,dataloader):
    '''
    The accuracy of the model on the verification set is calculated
    '''
    model.eval()
    val_losses, dcs = [], []
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input.cuda())
        val_label = Variable(label.cuda())
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
            model = model.cuda()
        concat = model_feature(val_input)
        
        outputs=model(concat)
        pred = outputs.data.max(1)[1].cpu().numpy().squeeze()
        gt = val_label.data.cpu().numpy().squeeze()

        if 1 > 0:
            dc,val_loss=calc_dice(gt[:,:,:],pred[:,:,:])
            dcs.append(dc)
            val_losses.append(val_loss)

    model.train()
    return np.mean(dcs),np.mean(val_losses)
############################################################################


############################################################################
print('train:')
lr = opt.lr
batch_size = batch_size.batch_size
print('batch_size:',batch_size,'lr:',lr)

plt_list = []

model = getattr(models, 'F_UNet_s2')() 
#model.load_state_dict(t.load('/userhome/xxxxxx.pth'))

check_path = 'userhome/xxxxxx/' # The path saves the corresponding model of feature engineering
checklist = os.listdir(check_path)
model_feature = getattr(models, 'F_UNet_s1')() 

if opt.use_gpu: 
    model.cuda()
    model_feature.cuda()
    
train_data=dataread(opt.train_data_root,train = True, test = False, val = False)
val_data=dataread(opt.val_data_root,train = False, test = False, val = True)

val_dataloader = DataLoader(val_data,1,shuffle=False,num_workers=opt.num_workers)
train_dataloader = DataLoader(train_data,batch_size = batch_size,shuffle=True,num_workers=opt.num_workers)

criterion = t.nn.CrossEntropyLoss()
#criterion = DiceLoss3D()

if opt.use_gpu: 
    criterion = criterion.cuda()

loss_meter=AverageMeter()
previous_loss = 1e+20

optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)

# train
for epoch in range(opt.max_epoch):
        
    loss_meter.reset()
    
    for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):

        # train model 
        input = Variable(data)
        target = Variable(label)
            
        if opt.use_gpu:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        
        ###
        ###
        ###
        checkpth = checklist[random.randint(0,len(checklist)-1)]
        model_feature.load_state_dict(t.load(check_path+checkpth))
        concat = model_feature(input)
        
        score = model(concat)

        loss = criterion(score,target)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())

        if ii==1:
            plt_list.append(loss_meter.val)
        if ii==1:
            print('train-loss-avg:', loss_meter.avg,'train-loss-each:', loss_meter.val)
           
        if ii==1:
            acc,val_loss = val(model,model_feature,val_dataloader)

            prefix = opt.pth_save_path + str(acc)+'_'+str(val_loss) + '_'+str(lr)+'_'+str(batch_size)+'_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            t.save(model.state_dict(), name)
            
            name1 = time.strftime(opt.loss_save_path + '%m%d_%H:%M:%S.npy')
            numpy.save(name1, plt_list)
            
    print('old:','batch_size:',batch_size,'lr:',lr)
    print('new:','batch_size:',batch_size,'lr:',lr)
############################################################################




