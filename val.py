import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs_ucm

from dataset1 import Dataset
from metrics import iou_score1,iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
import torch
from thop import profile

import shutil
import os


import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


import torch
import torch.nn as nn
def estimate_model_inference_memory_usage(model, val_loader, name ='ucm',device='cpu'):
    model.to(device)
    parameter_memory = 0

    # Calculate memory used by model parameters
    for param in model.parameters():
        parameter_memory += param.nelement() * param.element_size()

    # Estimate input tensor memory
   
    

    # Perform a forward pass to estimate output memory (without gradients)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=1):
            model.eval()
            input_memory = input.nelement() * input.element_size()
            if name== 'EGEUNet':
                pre,output = model(input)
                output_memory1 = sum([o.nelement() * o.element_size() for o in pre])
                output_memory = output.nelement() * output.element_size() +output_memory1
              #  iou,dice = iou_score1(output, target)
          
            else: 
                output = model(input,inference_mode=True)
               # iou,dice = iou_score(output, target)
                output = torch.sigmoid(output)
            break
        
       # output_tensor = model(input_tensor.to(device))
    output_memory = output.nelement() * output.element_size() 

    # Convert bytes to megabytes
    total_memory_MB = (parameter_memory + input_memory + output_memory) / (1024 ** 2)

    print(f"Estimated total memory usage during inference: {total_memory_MB:.4f} MB")

def fuse_conv_bn(conv, bn):
    """
    This function fuses a convolution layer with a batch normalization layer.
    
    Parameters:
    - conv (nn.Conv2d): The convolutional layer.
    - bn (nn.BatchNorm2d): The batch normalization layer.
    
    Returns:
    - nn.Conv2d: The fused convolutional layer.
    """
    # Step 1: Extract the parameters from BatchNorm
    bn_mean = bn.running_mean
    bn_var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    bn_weight = bn.weight
    bn_bias = bn.bias
    
    # Step 2: Adjust the Conv2D weight and bias
    conv_weight = conv.weight.clone().view(conv.out_channels, -1)
    conv_weight = bn_weight / bn_var_sqrt.view(-1, 1) * conv_weight
    conv_weight = conv_weight.view(conv.weight.size())
    conv_bias = bn_bias - bn_weight * bn_mean / bn_var_sqrt
    
    if conv.bias is not None:
        conv_bias += conv.bias
        
    # Step 3: Create a new Conv2D layer with the fused parameters
    fused_conv = nn.Conv2d(in_channels=conv.in_channels,
                           out_channels=conv.out_channels,
                           kernel_size=conv.kernel_size,
                           stride=conv.stride,
                           padding=conv.padding,
                           dilation=conv.dilation,
                           groups=conv.groups,
                           bias=True)
    fused_conv.weight = nn.Parameter(conv_weight)
    fused_conv.bias = nn.Parameter(conv_bias)
    
    return fused_conv

def fuse_model(model):
    """
    This function recursively fuses Conv2D and BatchNorm2D layers in the model.
    
    Parameters:
    - model (torch.nn.Module): The PyTorch model.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            # Check if the next layer is BatchNorm2D
            successor = next(model.named_children())[1]
            if isinstance(successor, nn.BatchNorm2d):
                # Fuse Conv2D and BatchNorm2D
                fused_conv = fuse_conv_bn(child, successor)
                setattr(model, child_name, fused_conv)
                # You might want to remove or replace the successor layer, e.g., with nn.Identity()
        else:
            # Recursively apply to children
            fuse_model(child)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    

    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--arch', default='transfuse', type=str,
                        help='model')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--path',default ='no', type = str)
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
    args = parser.parse_args()

    return args



def compute_gflops(model, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    config = vars(parse_args())
    input = torch.randn(1, 3, input_size, input_size)
    if config['arch'].strip() =='TransFuse_S':
        input = torch.randn(1, 3, 192,256)
    if config['arch'] =='TransUNet':
        input = torch.randn(1, 3, 224,224)
    #input = torch.randn(1, 3, 192,256) ## TransFuse_S
    print(input.shape,config['arch'] =='TransFuse_S',config['arch'])
    input = input.to(device)
    macs, params = profile(model, inputs=(input, ))
    gflops = macs / (10**9)
    return gflops


def main():
    config = vars(parse_args())
 
    print(config)

    if config['arch'] !='TransFuse_S':

        with open('models/%s/config.yml' % config['name'], 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    if config['arch']== 'EGEUNet':
        model = egeunet.EGEUNet()

        
    else:
        model = archs_ucm.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
     
    
    gflops = compute_gflops(model,config['input_h'])
    print(f'GigaFLOPs: {gflops}')
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
   
    if config['arch'] =='TransFuse_S':
        model = TransFuse_S(pretrained=True).cuda()

    
    


    
    
    
   

    
    val_img_ids = glob(os.path.join('', config['dataset'], 'val/','images/', '*' + config['img_ext']))

    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
   
 

    
    
 

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
      
       
    ])
  



    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('', config['dataset'], 'val/','images/'),
        mask_dir=os.path.join('', config['dataset'], 'val/','masks/'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform, train = False)


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,

        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
    pin_memory=True)

  
    model=torch.load('models/%s/model.pth' % config['name'])
    model.eval()
    model = model.cuda()


    
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    f1_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0

    preds = []
    gts = []
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            gts.append(target.squeeze(1).cpu().detach().numpy())
           # input, target = input.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).float()
            input = input.cuda()
            target = target.cuda()
            
            # compute output
            

            if config['arch']== 'EGEUNet':
                pre,output = model(input)
                iou,dice = iou_score1(output, target)

            else: 
                pre,output = model(input)
                iou,dice = iou_score(output, target)
                output = torch.sigmoid(output)



            iou_avg_meter.update(iou, input.size(0))
           
            dice_avg_meter.update(dice, input.size(0))
           
            
            output1 = output.squeeze(1).cpu().detach().numpy()
            preds.append(output1) 
            output = output.cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
            


    print('IoU: %.8f' % iou_avg_meter.avg)
    print('Dice: %.8f' % dice_avg_meter.avg)
    


    model=torch.load('models/%s/modelmiou1.pth' %
                                     config['name'])
    

    model.eval()
    model = model.cuda()



    count = 0

    preds = []
    gts = []
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            gts.append(target.squeeze(1).cpu().detach().numpy())
           # input, target = input.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).float()
            input = input.cuda()
            target = target.cuda()
            
            # compute output
            

            if config['arch']== 'EGEUNet':
                pre,output = model(input)
                iou,dice = iou_score1(output, target)
           
            else: 
                pre,output = model(input)
                iou,dice = iou_score(output, target)
                output = torch.sigmoid(output)
            '''
            iou_avg_meter.update(iou, input.size(0))
           
            dice_avg_meter.update(dice, input.size(0))
            '''
           
            
            output1 = output.squeeze(1).cpu().detach().numpy()
            preds.append(output1) 

                    
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
    print('miou*',miou)
    print('f1_or_dsc*',f1_or_dsc)
    
    
    
    

    model.eval()  # Set the model to evaluation mode
    #fuse_model(model) 

    # Measure the FPS
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    start_time = time.time()

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            #gts.append(target.squeeze(1).cpu().detach().numpy())
           # input, target = input.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).float()
            input = input.cuda()
            target = target.cuda()


            if config['arch']== 'EGEUNet':
                pre,output = model(input)
              #  iou,dice = iou_score1(output, target)

            else: 
                output = model(input,inference_mode=True)
               # iou,dice = iou_score(output, target)
                output = torch.sigmoid(output)


    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = len(val_loader) / elapsed_time

    print(f"FPS: {fps:.4f}")

    
    torch.cuda.empty_cache()
    estimate_model_inference_memory_usage(model,  val_loader,name = config['arch'],device='cpu')

    return iou_avg_meter.avg,dice_avg_meter.avg, miou,f1_or_dsc
if __name__ == '__main__':
    main()
