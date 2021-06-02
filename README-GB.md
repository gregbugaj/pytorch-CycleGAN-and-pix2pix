
```
    python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/patches --output_dir ./datasets/patches/ready

    python train.py --dataroot ./datasets/patches/ready --name maps_pix2pix --model pix2pix --direction AtoB --gpu_ids -1
```


