# UCM-Net

This is the official code repository for "UCM-Net: A Lightweight and Efficient Solution for Skin Lesion Segmentation using MLP and CNN"

**Prepare the dataset.**

- The PH2, ISIC2017 and ISIC2018 datasets, divided into a 7:3 ratio, can be found here {[One Drive]}. 
- {[PH2](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EWshLZcfAANIuH5Im4GJIJQB1bNBqQSVg6kLrL7MiLxWaQ?e=07ZwPp)}
- {[ISIC2017](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EW-0KZUjG5pLiLL1J0WmcmsBZd_Bn-AJq-vlny6ysHv7NQ?e=zPhqGs)}
- {[ISIC2018](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/ETR7uPI7hmdKhMnFAKr4MmcBuhovlxZW--hyoiJ85RbLkA?e=fsIxHm)}
-  


**Train the EGE-UNet.**
```
cd UCM-UNet
```
```
# PH2 
python train_newdata.py --dataset '../ph2/' --arch UCM_Net --name ph2_nb_testing_batch_8 --img_ext .bmp --mask_ext _lesion.bmp  --epochs 300 --loss GT_BceDiceLoss_new1 --batch_size 8
```
```
# ISIC2017 
python train_newdata.py --dataset '../isic2017/' --arch UCM_Net --name isic2017_nb_testing_batch_8 --img_ext .jpg --mask_ext _segmentation.png  --epochs 300 --loss GT_BceDiceLoss_new1 --batch_size 8

```
```
# ISIC2018
python train_newdata.py --dataset '../isic2018/' --arch UCM_Net --name isic2018_nb_testing_batch_8 --img_ext .png --mask_ext .png  --epochs 300 --loss GT_BceDiceLoss_new1 --batch_size 8

```

** Evaluate the model.**
```
# PH2
python val.py --name ph2_nb_testing_batch_8
# ISIC2017
python val.py --name isic2017_nb_testing_batch_8
# ISIC2018
python val.py --name isic2018_nb_testing_batch_8

```
