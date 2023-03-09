import os
from util.util import tensor2im
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import cv2
    

def process(segmap_path, target_path, i, phase, output_dir):

    w = 1024
    h = 1024

    segmap = cv2.imread(segmap_path)
    target = cv2.imread(target_path)


    save_phase = 'test' if phase == 'test' else 'train'
    savedir = os.path.join(output_dir, save_phase)

    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir+ 'A', exist_ok=True)
    os.makedirs(savedir+ 'B', exist_ok=True)

    # convert colorspace from OpenCV to PIL 
    segmap = cv2.cvtColor(segmap, cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    segmap = Image.fromarray(segmap)
    target = Image.fromarray(target)

    
    sidebyside = Image.new('RGB', (w * 2, h))
    sidebyside.paste(segmap, (w, 0))
    sidebyside.paste(target, (0, 0))
    
    savepath = os.path.join(savedir, "%d.png" % i)
    sidebyside.save(savepath, format='PNG', subsampling=0, quality=100)

    # data for cyclegan where the two images are stored at two distinct directories
    savepath = os.path.join(savedir + 'A', "%d_A.png" % i)
    target.save(savepath, format='PNG', subsampling=0, quality=100)
    savepath = os.path.join(savedir + 'B', "%d_B.png" % i)
    segmap.save(savepath, format='PNG', subsampling=0, quality=100)



class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        # self.update_realtime_augmentation(.02)

        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # copy single channel to 3 channels
        # R,G,B = AB.split()
        # img = Image.merge('RGB', (R,R,R))
        # # img.save('/tmp/pil/%s.png' %(index))
        # AB = img
        # w, h = AB.size
        # ima = Image.new('RGB', (w,h))
        # data = zip(AB.getdata(), AB.getdata(), AB.getdata())
        # ima.putdata(data)
        # AB = ima

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1),src=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1),src=False)

        A = A_transform(A)
        B = B_transform(B)
        
        # # So we need to reshape it to (H, W, C):
        # AA = A.permute(1,2,0)
        # npimg = AA.numpy()
        # print(AA.size())
        # image_numpy = tensor2im(AA)

        # aa = tensor2im(AA)
        # aa.save('/tmp/pil/%s_B.png' %(index))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
    

    def update_realtime_augmentation(self, augmentation_percentage=0.10):
        """Update the augmentation percentage for the current epoch.

        Parameters:
            augmentation_percentage (float) -- the augmentation percentage for the current epoch
        """
        import random

        augmentation_percentage = min(augmentation_percentage, 1.0)
        augmentation_percentage = max(augmentation_percentage, 0.0)

        # choose 10% of the images to apply augmentation to
        num_images = len(self.AB_paths)
        num_images_to_augment = int(num_images * augmentation_percentage)
        self.images_to_augment = set()

        output_dir = self.opt.dataroot
        data_dir = '/tmp/segmap'
        # output_dir = '/tmp/segmap/ready'

        image_dir = os.path.join(data_dir, "image")
        mask_dir = os.path.join(data_dir, "mask")

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        from layoutex.document_generator import DocumentGenerator
        from layoutex.layout_provider import LayoutProvider, get_layout_provider

        layout_provider = get_layout_provider("fixed", 10, 100, )
        generator = DocumentGenerator(
            layout_provider=layout_provider,
            target_size=1024 * 1,
            solidity=0.5,
            expected_components=["figure", "table"],
            assets_dir="~/dev/marieai/layoutex/assets"
        )

        # num_images_to_augment = 10
        # print(f'Starting process  {i} -> {phase}')

        for i in range(num_images_to_augment):            
            index = random.randint(0, num_images - 1)

            print(f"Generating image {i} of {num_images_to_augment}  [ {index} ]")

            document = generator.render(i)
            if not document.is_valid():
                continue

            image=document.image
            mask=document.mask

            segmap_path =  os.path.join(
                    mask_dir,
                    "blk_{}.png".format(str(index).zfill(8)),
            )
            
            image_path = os.path.join(
                    image_dir,
                    "blk_{}.png".format(str(index).zfill(8)),
            )
            
            image.save(image_path)
            mask.save(segmap_path)
            
            process(segmap_path, image_path, index, self.opt.phase , output_dir=output_dir)



