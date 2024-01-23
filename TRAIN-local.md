python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/claim_mask/src --output_dir ./datasets/claim_mask/ready


python train.py --dataroot ./datasets/claim_mask/ready --name claim_mask_local --model pix2pix --direction AtoB --gpu_ids 0 --batch_size 6 --netG local  --preprocess crop --display_freq 100 --lr 0.0002 --save_epoch_freq 1  --load_size 2048 --crop_size 512 --display_env claim_mask --no_flip --norm instance --netD pixel --n_epochs 100 --display_winsize 1280 --gan_mode ssim  --ndf 64 --ngf 64 --continue_train

python train.py --dataroot ./datasets/claim_mask/ready --name claim_mask_local --model pix2pix --direction AtoB --gpu_ids 0,1 --batch_size 24 --netG local  --preprocess crop --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --save_latest_freq 1500 --load_size 2048 --crop_size 256  --display_env claim_mask --no_flip --norm instance --netD n_layers_spectral --n_epochs 100 --display_winsize 1280  --gan_mode ssim  --ndf 64 --ngf 64 --continue_train


TESTING CYCLE GAN 
----------------------

python train.py --dataroot ./datasets/claim_mask/ready --name claim_mask_local --model cycle_gan --direction AtoB --gpu_ids 0 --batch_size 2 --netG local  --preprocess crop --display_freq 100 --lr 0.0002 --save_epoch_freq 1  --load_size 2048 --crop_size 256 --display_env claim_mask --no_flip --norm instance --netD basic --n_epochs 100 --display_winsize 1280 --gan_mode ssim  --ndf 64 --ngf 64 
