from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import numpy as np
import pickle
import random
import os, sys
import cv2
import PIL

_cur_dir = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
if _cur_dir not in sys.path:
    sys.path.append(_cur_dir)

def get_all_the_dirs_with_filter(path, filter="C"):
    list = os.listdir(path)
    dirs = []
    for dir in list:
        if dir.startswith(filter):
            dirs.append(os.path.join(path, dir))

    return dirs

def get_gt_labels_teeth(path, image_size):
    up = cv2.imread(os.path.join(path, "TeethEdgeUp.png"))
    up = cv2.resize(up, (image_size, image_size))
    down = cv2.imread(os.path.join(path, "TeethEdgeDown.png"))
    down = cv2.resize(down, (image_size, image_size))

    up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    up = np.expand_dims(((up > 10).astype(np.uint8)*255).squeeze(), -1)

    down = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY)
    down = np.expand_dims(((down > 10).astype(np.uint8)*255).squeeze(), -1)

    return np.concatenate([up, down], axis=-1) # H * W * 2

def get_gt_labels_mouth(path, image_size):

    mask = cv2.imread(os.path.join(path, "MouthMask.png"))
    mask = cv2.resize(mask, (image_size, image_size))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.expand_dims(((mask > 60).astype(np.uint8)*255).squeeze(), -1)

    return mask

def get_gt_image(path, image_size, blur=True):
    img = cv2.imread(os.path.join(path, "Img.jpg")).astype(np.uint8)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if blur and np.random.rand() > 0.5:
        min_kernel_size = 3
        max_kernel_size = 10
        size = np.random.randint(min_kernel_size, max_kernel_size)
        img = cv2.blur(img, ksize=(size, size))

    return img 

def preproc(opt):
    dataset_root = opt.dataset_root
    image_size = opt.image_size

    for mode in ("train", "valid"):

        data = {'x':[], 'y':[]}

        dirs = get_all_the_dirs_with_filter(os.path.join(dataset_root, mode))
        for i, d in enumerate(dirs):
            image = get_gt_image(d, image_size, blur=False)
            edge  = get_gt_labels_teeth(d, image_size)
            mask  = np.expand_dims(cv2.dilate(get_gt_labels_mouth(d, image_size), np.ones((7, 7), dtype=np.uint8)), -1)
            geom  = np.concatenate([edge, mask], axis=-1)

            data['x'].append(image)
            data['y'].append(geom)

            cnt = i + 1
            if cnt % 100 == 0:  print('[*] Preproc (%s): already %04d done' % (mode, cnt))

        print('[*] Preproc (%s): all images done, total data number: %04d' % (mode, cnt))

        with open('data/teeth-%s-%d.pkl' % (mode, opt.image_size), 'wb') as f:
            pickle.dump(data, f)
        print('[*] Preproc (%s): data saved at teeth-%s-%d.pkl' % (mode, mode, opt.image_size))
    return

class Teeth_Dataset(Dataset):
    def __init__(self, mode, opt):
        assert(mode == 'train' or mode == 'valid')
        self.mode = mode
        self.image_size = opt.image_size
        
        data_path = os.path.join('data', 'teeth-%s-%d.pkl' % (mode, opt.image_size))
        if not os.path.exists(data_path):
            preproc(opt)

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def postproc(self, x, y, degree=27, hflip_prob=0.5):
        x = TF.to_pil_image(x)
        y = TF.to_pil_image(y)
        
        if self.mode == 'train':
            # data augmentation
            x = TF.pad(x, (self.image_size // 4, self.image_size // 4), padding_mode='symmetric')
            y = TF.pad(y, (self.image_size // 4, self.image_size // 4), padding_mode='symmetric')
            
            angle = random.random()*degree - degree/2
            x = TF.rotate(x, angle, resample=PIL.Image.BILINEAR)
            y = TF.rotate(y, angle, resample=PIL.Image.BILINEAR)
            
            x = TF.center_crop(x, (self.image_size, self.image_size))
            y = TF.center_crop(y, (self.image_size, self.image_size))
            
            if random.random() < hflip_prob:
                x = TF.hflip(x)
                y = TF.hflip(y)
            
        x = TF.to_tensor(x)
        y = TF.to_tensor(y)
        y = (y > 0.6).float()
        
        return x, y

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, idx):
        x = self.data['x'][idx]     # ndarray(np.uint8) (H, W, C)   0~255
        y = self.data['y'][idx]     # ndarray(np.uint8) (H, W, C)   0, 255

        # ndarray->tensor & augmentation for training set
        x, y = self.postproc(x, y)
        
        return x, y

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset-root', type=str, default='D:/ImagePairs')
    arg_parser.add_argument('--image-size', type=int, default=256)
    opt = arg_parser.parse_args()

    # dataset = Teeth_Dataset('train', opt)
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=4,
    #     num_workers=1,
    #     shuffle=True
    # )
    # print(len(dataloader))
    # for x, y in dataloader:
    #     print(x.shape, y.shape)
    #     break

    dataset = Teeth_Dataset('valid', opt)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=1,
        shuffle=True
    )
    print(len(dataloader))
    for x, y in dataloader:
        print(x.shape, y.shape)
        break