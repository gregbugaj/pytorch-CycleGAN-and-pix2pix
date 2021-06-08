
```
    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/patches --output_dir ./datasets/patches/ready

    python train.py --dataroot ./datasets/patches/ready --name maps_pix2pix --model pix2pix --direction AtoB --gpu_ids -1
    python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1
    python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip

    python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 32 --epoch

```

### Continue training

```
python train.py --dataroot ./datasets/patches/ready --name form_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 64 --continue_train --n_epochs 5000 --display_freq 100
```

