
import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
import albumentations as A
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize,Rotate, VerticalFlip,HorizontalFlip, ElasticTransform
import archs_ucm
import losses
from dataset1 import Dataset
from metrics import iou_score,iou_score1
from utils import AverageMeter, str2bool


import numpy as np
from sklearn.metrics import confusion_matrix

from torch.nn.modules.loss import CrossEntropyLoss

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
ARCH_NAMES = archs_ucm.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='AdamW',
                        choices=['AdamW', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['AdamW', 'SGD']) +
                        ' (default: SGD)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--nrand', default=44, type=int,
                        help='rand state')
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--num_workers', default=4, type=int)
  #  parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file' )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'new_iou':AverageMeter()}

    model.train()
    ce_loss = CrossEntropyLoss()
   

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['arch']== 'TransUNet':
            outputs = model(input)

            
            loss_dice = dice_loss(outputs, target)
            
            
            loss_ce = ce_loss(outputs, target)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = loss.mean()
            iou,dice = iou_score(outputs, target)
            
        elif config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        elif config['loss'] =='LogNLLLoss':
            #gts.append(target.squeeze(1).cpu().detach().numpy())
            output = model(input)
            tmp2 = target.detach().cpu().numpy()
            tmp = output.detach().cpu().numpy()
            tmp[tmp>=0.5] = 1
            tmp[tmp<0.5] = 0
            tmp2[tmp2>0] = 1
            tmp2[tmp2<=0] = 0
            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)

            yHaT = tmp
            yval = tmp2



            loss = criterion(output, target)
            loss =loss.mean()

            iou,dice = iou_score(output, target)
         #   output1 = torch.sigmoid(output)
         #   output1 = output1.squeeze(1).cpu().detach().numpy()
         #   preds.append(output1)
        elif config['loss'] =='GT_BceDiceLoss':
            gt_pre, out = model(input)
         
            loss = 0
    

            loss = criterion(gt_pre, out, target)
            loss = loss.mean()

         
            iou,dice = iou_score(out, target)
        elif config['loss'] =='GT_BceDiceLoss_new':
            gt_pre, out = model(input)
         
            loss = 0
    

            loss = criterion(gt_pre, out, target)
            loss = loss.mean()

         
            iou,dice = iou_score(out, target)
        elif config['loss'] =='GT_BceDiceLoss_new1':
            gt_pre, out = model(input)
         
            loss = 0
    

            loss,loss1,loss2,los3,loss4,loss5 = criterion(gt_pre, out, target)
 
            loss = loss.mean()
          #  loss1 = loss.mean()
           # loss2 = loss.mean()
           # loss3 = loss.mean()
           # loss4 = loss.mean()
           # loss5 = loss.mean()

         
            iou,dice = iou_score(out, target)
        elif config['loss'] =='BCEDiceLossMAL':
            out = model(input)
    
            loss = 0
    

            loss = criterion(out, target)
            loss = loss.mean()

   
            iou,dice = iou_score1(out, target)    
        elif config['loss'] =='BCEDiceLossUNEXT'or config['arch'] =='AttU_Net'or config['arch'] == 'R2U_Net' or config['arch'] =='U_Net':
            out = model(input)
      
            loss = 0
   

            loss = criterion(out, target)
            loss = loss.mean()

   
            iou,dice = iou_score(out, target)
        elif config['loss'] =='BCEDiceLossSWIN':
            out = model(input)
      
            loss = 0
   

            loss = criterion(out, target)
            loss = loss.mean()

   
            iou,dice = iou_score(out, target)
    
        elif config['loss'] =='LossTransFuse':
            lateral_map_4, lateral_map_3, lateral_map_2 = model(input)

            # ---- loss function ----
            loss4 = criterion(lateral_map_4, target)
            loss3 = criterion(lateral_map_3, target)
            loss2 = criterion(lateral_map_2, target)

            loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

   
            loss =loss.mean()
            iou,dice = iou_score(lateral_map_2, target)

    
        elif config['loss'] =='GT_BceDiceLossEGE':
            gt_pre, out = model(input)
            loss = 0
    

            loss = criterion(gt_pre, out, target)
            loss =loss.mean()

          
            iou,dice = iou_score1(out, target)

        elif config['arch']== 'CONVUNext':
            output = model(input)
            loss = 0
    

            loss = criterion(output['out'], target)
            loss =loss.mean()
          
            iou,dice = iou_score(output['out'], target)
        elif config['loss'] =='BCEDiceLossMEDT':
            output = model(input)
   



            loss = criterion(output, target)
            loss =loss.mean()
          
            iou,dice = iou_score(output, target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)
            
        if config['loss'] !='GT_BceDiceLoss_new1':
        # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
           # print('hello')
            # Zero gradients
            optimizer.zero_grad()

            # Backward pass for the final output loss
            loss.backward(retain_graph=True)

            # Backward pass for each internal layer loss
            
            
            
            
           # loss1.backward(retain_graph=True)
            #loss2.backward(retain_graph=True)
           # loss3.backward(retain_graph=True)
           # loss4.backward(retain_graph=True)
          #  loss5.backward(retain_graph=True)
            # Optional: Modify gradients here if needed

            # Update parameters
            optimizer.step()
            

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

best_miou1 = 0
best_dice1 = 0
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    
    preds = []
    gts = []
    ce_loss = CrossEntropyLoss()
 
    with torch.no_grad():

        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            
            
            input = input.cuda()
            target = target.cuda()

    

            if config['loss'] =='GT_BceDiceLoss' or config['loss'] =='GT_BceDiceLoss_new':
                gts.append(target.squeeze(1).cpu().detach().numpy())
                
                gt_pre, out = model(input)
            
                loss = 0
              

                loss = criterion(gt_pre, out, target)
                loss = loss.mean()

      
                iou,dice = iou_score(out, target)
                
                output1 = torch.sigmoid(out)
                output1 = output1.squeeze(1).cpu().detach().numpy()

                preds.append(output1) 
            elif config['loss'] =='GT_BceDiceLoss_new1':
                gts.append(target.squeeze(1).cpu().detach().numpy())
                
                
                gt_pre, out = model(input)

                loss = 0


                loss,loss1,loss2,los3,loss4,loss5 = criterion(gt_pre, out, target)
                iou,dice = iou_score(out, target)
                
                output1 = torch.sigmoid(out)
                output1 = output1.squeeze(1).cpu().detach().numpy()

                preds.append(output1) 
            
           
            
            elif config['loss'] =='BCEDiceLossUNEXT' or config['arch'] =='AttU_Net'or config['arch'] == 'R2U_Net' or config['arch'] =='U_Net':
                gts.append(target.squeeze(1).cpu().detach().numpy())
                out = model(input)
           
                loss = 0
              

                loss = criterion(out, target)
                loss = loss.mean()

         
                iou,dice = iou_score(out, target)
                output1 = torch.sigmoid(out)
                output1 = output1.squeeze(1).cpu().detach().numpy()
                preds.append(output1)



            elif config['loss'] =='GT_BceDiceLossEGE':
                gts.append(target.squeeze(1).cpu().detach().numpy())
                gt_pre, out = model(input)
                loss = 0
          

                loss = criterion(gt_pre, out, target)
                loss =loss.mean()
              
   
                iou,dice = iou_score1(out, target)
                out= out.squeeze(1).cpu().detach().numpy()
                preds.append(out) 

            else:
                output = model(input)
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        #print(preds)

        y_pre = np.where(preds>=0.5, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        global best_miou1
        global best_dice1
        if best_miou1<miou:
            torch.save(model, 'models/%s/modelmiou1.pth' %
                       config['name'])
            best_miou1 = miou
            best_dice1 = f1_or_dsc
            
        print('miou',best_miou1)
        print('f1_or_dsc',best_dice1)
        pbar.close()

    
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])
def model_summary(model):
    print(model)  # This will print the architecture of the model

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total Parameters: {total_params}')
    print(f'Trainable Parameters: {trainable_params}')

import torch
from thop import profile

def compute_gflops(model, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    config = vars(parse_args())
    input = torch.randn(1, 3, input_size, input_size)
    if config['arch'] =='TransFuse_S':
        input = torch.randn(1, 3, 192,256)
    if config['arch'] =='TransUNet':
        input = torch.randn(1, 3, 224,224)
    input = input.to(device)
    macs, params = profile(model, inputs=(input, ))
    gflops = macs / (10**9)
    return gflops


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    # create model
    model = archs_ucm.__dict__[config['arch']](config['num_classes'],
                                       config['input_channels'],
                                       config['deep_supervision'])   
    config['optimizer'] == 'AdamW'
    weight_decay=0.01
    config['scheduler'] == 'CosineAnnealingLR'
    T_max=50#config['epochs']
        
        

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True


    model_summary(model)#

    input_size = config['input_h']  # Set this to the height/width of your input images
    gflops = compute_gflops(model, input_size)
    print(f'GigaFLOPs: {gflops}')
   # model = torch.load('models/%s/model.pth' %
                     #  config['name'])
    model = model.cuda()
    if config['arch'] =='SWIN':
        model.load_from(config1)
        
    params = filter(lambda p: p.requires_grad, model.parameters())

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],weight_decay=config['weight_decay'])
                           #   nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    train_img_ids = glob(os.path.join('', config['dataset'], 'train/','images/', '*' + config['img_ext']))

    val_img_ids = glob(os.path.join('', config['dataset'], 'val/','images/', '*' + config['img_ext']))

 
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
 

    
    
    train_transform = Compose([
        Rotate(limit=180,p=0.5),
        VerticalFlip(p =0.5),HorizontalFlip(p =0.5),
        Resize(config['input_h'], config['input_w']),

    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),

    ])
    print('train_img_ids',len(train_img_ids))
    print('val_img_ids',len(val_img_ids))

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('', config['dataset'], 'train/','images/'),
        mask_dir=os.path.join('', config['dataset'], 'train/','masks/'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform, train = True)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('', config['dataset'], 'val/','images/'),
        mask_dir=os.path.join('', config['dataset'], 'val/','masks/'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform, train = False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,

        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        
       
        scheduler.step()


        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model, 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0
        print('best_iou', best_iou)
        print('best_dice', best_dice)

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

