import os
import glob
from PIL import Image, ImageOps

help_msg = """
The processed images will be placed at --output_dir.
Example usage:

python ./prepare_patches_dataset.py  --input_dir ../datasets/patches --output_dir ../datasets/patches/ready

"""

def load_resized_img(path, size):
    return Image.open(path).convert('RGB').resize(size)
    
def process(input_dir, output_dir, phase):
    save_phase = 'test' if phase == 'test' else 'train'

    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + 'A', exist_ok=True)
    os.makedirs(savedir + 'B', exist_ok=True)

    print("Directory structure prepared at %s" % output_dir)
    
    segmap_expr = os.path.join(input_dir, phase) + "/mask/*.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    photo_expr = os.path.join(input_dir, phase) + "/image/*.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    # filter and only get th

    assert len(segmap_paths) == len(photo_paths), \
        "%d images that match [%s], and %d images that match [%s]. Aborting." % (len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)

    assert len(segmap_paths),"No Images found in the directory. Aborting."

    w = 700
    h = 350    

    # HICFA    
    w = 1024
    h = 1024

    w = 1000
    h = 256    
    
    # diagnosis_code
    # w = 1680
    # h = 220

    print(len(segmap_paths))
    for i, (segmap_path, photo_path) in enumerate(zip(segmap_paths, photo_paths)):
        
        segmap = load_resized_img(segmap_path, (w, h))
        segmap = ImageOps.invert(segmap)
        photo = load_resized_img(photo_path, (w, h))

        # h = segmap.size[1]
        # w = segmap.size[0]
        # data for pix2pix where the two images are placed side-by-side
        # sidebyside = Image.new('RGB', (512, 256))
        # sidebyside.paste(segmap, (256, 0))
        # sidebyside.paste(photo, (0, 0))
        
        sidebyside = Image.new('RGB', (w * 2, h))
        sidebyside.paste(segmap, (w, 0))
        sidebyside.paste(photo, (0, 0))
        
        savepath = os.path.join(savedir, "%d.jpg" % i)
        sidebyside.save(savepath, format='JPEG', subsampling=0, quality=100)

        # data for cyclegan where the two images are stored at two distinct directories
        savepath = os.path.join(savedir + 'A', "%d_A.jpg" % i)
        photo.save(savepath, format='JPEG', subsampling=0, quality=100)
        savepath = os.path.join(savedir + 'B', "%d_B.jpg" % i)
        segmap.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        if i % (len(segmap_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(segmap_paths), savepath))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='./datasets/patches/ready',
                        help='Directory the output images will be written to.')
    opt = parser.parse_args()

    print(help_msg)
    
    print('Preparing Dataset for train phase')
    process(opt.input_dir, opt.output_dir, "train")

    print('Preparing Dataset for val phase')
    process(opt.input_dir, opt.output_dir, "test")

    print('Done')

    

