
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

python train.py --dataroot ./datasets/box31/ready --name box31_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 8 --display_freq 100  --netG resnet_9blocks  --preprocess none

python test.py --dataroot ./datasets/box31/eval --name box31_pix2pix --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --preprocess none
```

## BOX 33 Hyperparams

```
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/box33 --output_dir ./datasets/box33/ready

python train.py --dataroot ./datasets/box33/ready --name box33_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 4 --display_freq 100  --netG resnet_9blocks  --preprocess none

python test.py --dataroot ./datasets/box33/eval --name box33_pix2pix --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --preprocess none  --no_dropout
```


## Diagnosis code Hyperparams

```
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/diagnosis_code --output_dir ./datasets/diagnosis_code/ready

python train.py --dataroot ./datasets/diagnosis_code/ready --name diagnosis_code --model pix2pix --direction AtoB --gpu_ids 1 --no_flip --batch_size 4 --display_freq 100  --netG resnet_9blocks  --preprocess none --save_latest_freq 2000 --save_epoch_freq 1 --display_env diagnosis_code

python test.py --dataroot ./datasets/diagnosis_code/eval --name diagnosis_code --model test --netG resnet_9blocks --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --preprocess none
```

## Unet 1024

## HICFA Segmentation Hyperparams
```
python train.py --dataroot ./datasets/hicfa/ready --name hicfa_pix2pix --model pix2pix --direction AtoB --gpu_ids 0 --no_flip --batch_size 4 --display_freq 100  --netG unet_256  --load_size 1024 --crop_size 1024 --save_latest_freq 2000 --save_epoch_freq 1 --lr .0002 --display_env hicfa

python test.py --dataroot ./datasets/hicfa/eval_1024 --name hicfa_pix2pix --model test --netG unet_256 --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --load_size 1024 --crop_size 1024
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