
```
    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/forms/ --output_dir ./datasets/forms/ready

    python train.py --dataroot ./datasets/patches/ready --name maps_pix2pix --model pix2pix --direction AtoB --gpu_ids -1
    python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1
    python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip

    python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 32 --epoch    
    
    
    python train.py --dataroot ./datasets/forms/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0 --no_flip --batch_size 32 --epoch


python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 16 --n_epochs 500 --display_freq 100  --netG resnet_9blocks


python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 16 --n_epochs 500 --display_freq 100  --netG resnet_9blocks --serial_batches --load_size 512 --crop_size 512


python test.py --dataroot ./datasets/eval/0001 --name form_pix2pix --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch --load_size 340 --crop_size 340


python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 16 --n_epochs 500 --display_freq 100  --netG resnet_9blocks

```
Testing different network

```
python train.py --dataroot ./datasets/forms/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0 --no_flip --batch_size 8 --netG resnet_9blocks

```
### Continue training

```
python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 64 --continue_train --n_epochs 500 --display_freq 100
```


## BOX 31 Hyperparams
```
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/box31 --output_dir ./datasets/box31/ready

python train.py --dataroot ./datasets/box31/ready --name box31_pix2pix --model pix2pix --direction AtoB --gpu_ids 1 --no_flip --batch_size 8 --display_freq 100  --netG resnet_9blocks  --preprocess none --output_nc 1 --input_nc 1

python test.py --dataroot ./datasets/box31/eval --name box31_pix2pix --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --preprocess none --output_nc 1 --input_nc 1
```

## BOX 33 Hyperparams

```
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/box33 --output_dir ./datasets/box33/ready

python train.py --dataroot ./datasets/box33/ready --name box33_pix2pix --model pix2pix --direction AtoB --gpu_ids 1 --no_flip --batch_size 24 --display_freq 100  --netG resnet_9blocks  --preprocess crop --load_size 1000 --crop_size 228 --save_latest_freq 2000 --save_epoch_freq 1 --lr .0002  --norm instance   --output_nc 1 --input_nc 1 --display_env box33 


python test.py --dataroot ./datasets/box33/eval_1200/ --name box33_pix2pix --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1  --preprocess none --norm instance --output_nc 1 --input_nc 1

```
anna bugaj bolt
## Box 33 Hyperparams UNET

```
    UNET-256

    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/box33 --output_dir ./datasets/box33/ready_unet

    python train.py --dataroot ./datasets/box33/ready/ --name box33_unet_256 --model pix2pix --direction AtoB --gpu_ids 0,1  --batch_size 128 --netG unet_256  --preprocess scale_width_and_crop  --display_freq 100 --lr 0.0001 --save_epoch_freq 1 --load_size 1024  --crop_size 256 --output_nc 1 --input_nc 1 --no_flip --save_epoch_freq 1  --save_latest_freq 4000  --lambda_L1 100 --dataset_mode aligned --norm instance --pool_size 0 --continue_train

    python test.py --dataroot ./datasets/box33/eval_1024/ --name box33_unet_256 --model test --netG unet_256 --direction AtoB --dataset_mode single --gpu_id -1 --preprocess none --output_nc 1 --input_nc 1   --norm instance 

```

## Diagnosis code Hyperparams

```
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/diagnosis_code --output_dir ./datasets/diagnosis_code/ready

python train.py --dataroot ./datasets/diagnosis_code/ready --name diagnosis_code --model pix2pix --direction AtoB --gpu_ids 1 --no_flip --batch_size 4 --display_freq 100  --netG resnet_9blocks  --preprocess none --save_latest_freq 2000 --save_epoch_freq 1 --display_env diagnosis_code

python test.py --dataroot ./datasets/diagnosis_code/eval --name diagnosis_code --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --preprocess none
```


## HCFA02 Patient Name  Hyperparameters
```
    w = 1000
    h = 160

    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/HCFA02 --output_dir ./datasets/HCFA02/ready


 python train.py --dataroot ./datasets/HCFA02/ready --name HCFA02 --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 24 --netG unet_pp  --preprocess  scale_width_and_crop  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1024  --crop_size 192 --output_nc 1 --input_nc 3  --save_latest_freq 4000  --norm instance --netD n_layers --n_layers_D 5 --gan_mode focus --lr_policy plateau 

    python test.py --dataroot ./datasets/HCFA02/eval --name HCFA02 --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --preprocess none --output_nc 1 --input_nc 1

UNET PLUS PLUS

 
python train.py --dataroot ./datasets/diagnosis_code/ready --name diagnosis_code --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 16 --netG unet_pp  --preprocess  scale_width_and_crop  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1680  --crop_size 192 --output_nc 1 --input_nc 3  --save_latest_freq 4000  --norm instance --netD n_layers --n_layers_D 5 --lr_policy linear --no_dropout  --continue_train


python test.py --dataroot ./datasets/diagnosis_code/eval --name diagnosis_code --model test --netG unet_pp --direction AtoB --dataset_mode single --gpu_id -1 --norm instance  --preprocess none --output_nc 1 --input_nc 3 --no_dropout
 
    python ./datasets/prepare_patches_dataset.py  --input_dir /home/greg/dev/unet-denoiser/data-HCFA02-SET-2/ --output_dir ./datasets/HCFA02/ready

```

TRAINING INFO
https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198

## HCFA05PatientAddressOne  Hyperparameters
```
    w = 800
    h = 140

    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/HCFA05PatientAddressOne --output_dir ./datasets/HCFA05PatientAddressOne/ready
    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/HCFA05PatientAddressOne --output_dir ./datasets/HCFA05PatientAddressOne/ready_unet

    python train.py --dataroot ./datasets/HCFA05PatientAddressOne/ready --name HCFA05PatientAddressOne --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 16 --netG resnet_9blocks  --preprocess none --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 800  --output_nc 1 --input_nc 1 --norm instance --save_epoch_freq 1 --save_latest_freq 5000 --continue_train --epoch 3

    python train.py --dataroot ./datasets/HCFA05PatientAddressOne/ready --name HCFA05PatientAddressOne --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --netG resnet_9blocks --display_freq 100 --lr 0.0003 --save_epoch_freq 1  --output_nc 1 --input_nc 1 --norm instance   --save_epoch_freq 1  --save_latest_freq 4000 --batch_size 64 --preprocess crop --crop_size 140 --load_size 800 --pool_size 0 --continue_train --epoch 1


    python test.py --dataroot ./datasets/HCFA05PatientAddressOne/eval --name HCFA05PatientAddressOne --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1  --preprocess none --output_nc 1 --input_nc 1 --norm instance 

    python train.py --dataroot ./datasets/HCFA05PatientAddressOne/ready_unet/ --name HCFA05PatientAddressOne_unet_256 --model pix2pix --direction AtoB --gpu_ids 0,1  --batch_size 24 --netG unet_256  --preprocess scale_width_and_crop  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1024  --crop_size 256 --output_nc 1 --input_nc 1 --no_flip --save_epoch_freq 1  --save_latest_freq 4000  --lambda_L1 100 --dataset_mode aligned --norm instance --pool_size 0 --continue_train

    python test.py --dataroot ./datasets/HCFA05PatientAddressOne/eval_unet_256 --name HCFA05PatientAddressOne_unet_256 --model test --netG unet_256 --direction AtoB --dataset_mode single --gpu_id -1 --preprocess none --output_nc 1 --input_nc 1 --norm instance  


    UNET PLUS PLUS

    python train.py --dataroot ./datasets/HCFA05PatientAddressOne/ready_unet/ --name HCFA05PatientAddressOne_unet_pp --model pix2pix --direction AtoB --gpu_ids 0,1  --batch_size 8 --netG unet_pp  --preprocess scale_width_and_crop  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1024  --crop_size 1024 --output_nc 1 --input_nc 1 --no_flip --save_epoch_freq 1  --save_latest_freq 4000  --lambda_L1 100 --dataset_mode aligned --norm batch  

    python test.py --dataroot ./datasets/HCFA05PatientAddressOne/eval_unet --name HCFA05PatientAddressOne_unet_pp --model test --netG unet_pp --direction AtoB --dataset_mode single --gpu_id -1 --preprocess none --output_nc 1 --input_nc 1 --norm instance  

``'
b ea


## CYCLE GAN 
```
python train.py --dataroot ./datasets/HCFA05PatientAddressOne/ready --name HCFA05PatientAddressOne_cycle_gan --model cycle_gan --direction AtoB --gpu_ids 0 --no_flip --batch_size 1 --netG resnet_9blocks  --display_freq 100 --lr 0.0002 --save_epoch_freq 1  --output_nc 1 --input_nc 1 --norm instance   --save_epoch_freq 1  --save_latest_freq 1000  --preprocess scale_width --crop_size 400 --load_size 800

```

## HCFA07Phone Hyperparameters
```
    w = 600
    h = 140

    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/HCFA07Phone --output_dir ./datasets/HCFA07Phone/ready_unet

    python train.py --dataroot ./datasets/HCFA07Phone/ready --name HCFA07Phone --model pix2pix --direction AtoB --gpu_ids 0  --batch_size 6 --netG resnet_9blocks  --preprocess none --display_freq 100 --lr 0.0003 --save_epoch_freq 1 --load_size 600 --crop_size 140 --output_nc 1 --input_nc 1  --norm instance  --no_dropout  --no_flip --save_epoch_freq 1  --save_latest_freq 1000  --continue_train

    python test.py --dataroot ./datasets/HCFA07Phone/eval --name HCFA07Phone --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --preprocess none --output_nc 1 --input_nc 1 --norm instance


    Unet_512

    python train.py --dataroot ./datasets/HCFA07Phone/ready_unet/ --name HCFA07Phone --model pix2pix --direction AtoB --gpu_ids 0  --batch_size 8 --netG unet_512  --preprocess none  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 512  --crop_size 512 --output_nc 1 --input_nc 1  --norm instance  --no_dropout  --no_flip --save_epoch_freq 1  --save_latest_freq 1000 

    python test.py --dataroot ./datasets/HCFA07Phone/eval_unet_256/ --name HCFA07Phone --model test --netG unet_512 --direction AtoB --dataset_mode single --gpu_id -1 --preprocess none --output_nc 1 --input_nc 1 --load_size 512  --crop_size 512 --output_nc 1 --input_nc 1  --norm instance  --no_dropout

    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/HCFA07Phone --output_dir ./datasets/HCFA07Phone/ready_unet_128

    python train.py --dataroot ./datasets/HCFA07Phone/ready_unet_256/ --name HCFA07Phone_unet_256_pix2pix --model pix2pix --direction AtoB --gpu_ids 0  --batch_size 24 --netG unet_256  --preprocess scale_width_and_crop  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 512  --crop_size 256 --output_nc 1 --input_nc 1 --no_flip --save_epoch_freq 1  --save_latest_freq 4000  --lambda_L1 100 --dataset_mode aligned --norm instance --pool_size 0 --continue_train

    python test.py --dataroot ./datasets/HCFA07Phone/eval_unet_256 --name HCFA07Phone_unet_256_pix2pix --model test --netG unet_256 --direction AtoB --dataset_mode single --gpu_id -1 --preprocess none --output_nc 1 --input_nc 1 --norm instance  

    Final  model
    
    python train.py --dataroot ./datasets/HCFA07Phone/ready/ --name HCFA07Phone_resnet_9blocks_pix2pix --model pix2pix --direction AtoB --gpu_ids 0  --batch_size 24 --netG resnet_9blocks  --preprocess scale_width_and_crop  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 512  --crop_size 148 --output_nc 1 --input_nc 1 --no_flip --save_epoch_freq 1  --save_latest_freq 4000  --lambda_L1 100 --dataset_mode aligned --norm instance --pool_size 0

    python test.py --dataroot ./datasets/HCFA07Phone/eval_unet_256 --name HCFA07Phone_resnet_9blocks_pix2pix --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --preprocess none --output_nc 1 --input_nc 1 --norm instance  

```

## Serice Line  Hyperparams
```
izabella
UNET-256
2560 x 1024
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/service_lines --output_dir ./datasets/service_lines/ready


python train.py --dataroot ./datasets/service_lines/ready/ --name service_lines_unet --model pix2pix --direction AtoB --gpu_ids 0,1  --batch_size 128 --netG unet_256  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --preprocess scale_width_and_crop  --load_size 2560  --crop_size 256 --output_nc 1 --input_nc 1 --no_flip --save_epoch_freq 5  --save_latest_freq 4000  --lambda_L1 100 --dataset_mode aligned --norm instance --pool_size 0 --n_epochs 2000 --continue_train


 python test.py --dataroot ./datasets/service_lines/eval --name service_lines_unet --model test --netG unet_128 --direction AtoB --dataset_mode single --gpu_id -1 --preprocess none --output_nc 1 --input_nc 1 --norm instance  

UNET-128

python train.py --dataroot ./datasets/service_lines/ready/ --name service_lines_unet --model pix2pix --direction AtoB --gpu_ids 0,1  --batch_size 400 --netG unet_128  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --preprocess crop  --load_size 2560  --crop_size 128 --output_nc 1 --input_nc 1 --no_flip --save_epoch_freq 1 --save_latest_freq 24000  --lambda_L1 100 --dataset_mode aligned --norm instance --pool_size 0 --n_epochs 2000 --continue_train

```


## Unet 1024

## HICFA Segmentation Hyperparams
```

python test.py --dataroot ./datasets/hicfa/eval_1024 --name hicfa_pix2pix --model test --netG unet_512 --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --load_size 1024 --crop_size 512

python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/hicfa --output_dir ./datasets/hicfa/ready

-- all fields

python train.py --dataroot ./datasets/hicfa/ready --name hicfa_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 6 --display_freq 100  --preprocess none  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1024  --crop_size 1024 --output_nc 3 --input_nc 3 --save_epoch_freq 1  --save_latest_freq 2000 --save_epoch_freq 1 --lr .0002 --display_env hicfa --n_epochs 300 --netG unet_1024 

python test.py --dataroot ./datasets/hicfa/eval_1024 --name hicfa_pix2pix --model test --netG unet_1024 --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --load_size 1024 --crop_size 1024


python ./datasets/prepare_patches_dataset.py  --input_dir /home/greg/dev/assets-private/cvat/TRAINING-ON-DD-GPU/hicfa-forms/output_split --output_dir ./datasets/hicfa/ready

```

UNET PP

```
python train.py --dataroot ./datasets/hicfa/ready_unet/ --name HCFA05PatientAddressOne_unet_pp --model pix2pix --direction AtoB --gpu_ids 0,1  --batch_size 8 --netG unet_pp    --lambda_L1 100 --dataset_mode aligned --norm batch

python test.py --dataroot ./datasets/hicfa/eval_1024 --name hicfa_pix2pix_pp --model test --netG unet_pp --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --load_size 1024 --crop_size 1024

```



## Diagnosis code Hyperparameter

```

python train.py --dataroot ./datasets/diagnosis_code/ready --name diagnosis_code --model pix2pix --direction AtoB --gpu_ids 0 --no_flip --batch_size 2 --netG resnet_9blocks  --preprocess crop --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1600 --crop_size 228 --display_env diagnosis


python test.py --dataroot ./datasets/diagnosis_code/eval --name diagnosis_code --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm instance  --preprocess none --no_dropout --norm instance


```
## Dataset

```
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/box33 --output_dir ./datasets/box33/ready
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/hicfa --output_dir ./datasets/hicfa/ready

    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/diagnosis_code --output_dir ./datasets/diagnosis_code/ready

```

python test.py --dataroot ./datasets/box33/eval_1024 --name box33_pix2pix --model test --netG unet_256 --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --load_size 1024 --crop_size 1024



# Ref
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/325


python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/service_lines_im/ --output_dir ./datasets/service_lines_im/ready


## HICFA MASK - New process

python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/hicfa_mask/src --output_dir ./datasets/hicfa_mask/ready

python train.py --dataroot ./datasets/hicfa_mask/ready --name hicfa_mask --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 8 --netG unet_256  --preprocess crop --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1700 --crop_size 256 --display_env hicfa_mask --no_dropout --norm instance


python test.py --dataroot /tmp/form-segmentation/mask --name hicfa_mask --model test --netG unet_256_spectral --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --load_size 1700 --preprocess none


# SPECTRAL

python train.py --dataroot ./datasets/hicfa_mask/ready --name hicfa_mask --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 8 --netG unet_256_spectral  --preprocess crop --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1700 --crop_size 256 --display_env hicfa_mask --no_dropout --norm instance --netD n_layers_spectral



python test.py --dataroot ./datasets/hicfa_mask/eval --name hicfa_mask --model test --netG unet_256_spectral --direction AtoB --dataset_mode single --gpu_id -1 --norm instance  --preprocess none --no_dropout 
