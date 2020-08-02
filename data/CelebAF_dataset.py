from torch.utils.data import Dataset
from threading import Thread
from queue import Queue
import torchvision.transforms.functional as TF
import numpy as np
import torch
import random
import os, shutil, sys
import cv2
import PIL

_cur_dir = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
if _cur_dir not in sys.path:
    sys.path.append(_cur_dir)

from __init__ import detector, predictor, get_pose

def resize_to_image_size(image, image_size=256):
    H, W, _ = image.shape

    if H != image_size or W != image_size:
        if W < H:
            image = cv2.resize(image, (image_size, int(H*image_size/W)))
            start = int(image.shape[0] - image_size) // 2
            image = image[start:start+image_size, :, :]
        else:
            image = cv2.resize(image, (int(W*image_size/W), image_size))
            start = int(image.shape[1] - image_size) // 2
            image = image[:, start:start+image_size, :]

    return image

def face_extractor(img, det, p2d):
    y0 = det.top()
    pts = np.empty([19, 2])
    for j in range(19):
        if j == 0:
            pts[j] = [p2d[0, 0], y0]
        elif j == 18:
            pts[j] = [p2d[16,0], y0]
        else:
            # 1 ~ 17: jaw line
            pts[j] = p2d[j-1]

    mask = np.zeros_like(img[:,:,0])    # (H, W)
    cv2.fillPoly(mask, [pts.astype(np.int)], 1)
    mask = mask.astype(np.uint8)*255
    return mask

def proc_single_file(file_name, threshold=4., debug=False):
    image = cv2.imread(file_name)
    image = resize_to_image_size(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    if len(dets) == 1:
        det = dets[0]

        shape = predictor(image, det)
        p2d = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(shape.num_parts)], np.float32)

        mask = face_extractor(image, det, p2d)
        
        theta = get_pose(det, p2d)
        chosen = (theta > 0 and theta < threshold)
    else:
        chosen, det, p2d, mask, theta = False, None, None, None, -1
        
    if debug:
        return chosen, image, det, p2d, mask, theta
    else:
        return chosen, image, det, p2d, mask

def gen_dataset(raw_dataset_root, tgt_dataset_root):
    files = os.listdir(os.path.join(raw_dataset_root, 'Img'))
    cnt = 0
    for n, f in enumerate(files):
        chosen, image, bbox, landmark, mask = proc_single_file(os.path.join(raw_dataset_root, 'Img', f))

        if chosen:
            assert(landmark is not None)
            x = np.copy(image)
            geom = landmark.astype(np.int)
            shap = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
            mode = 'train' if random.random() < 0.8 else 'valid'
            os.makedirs(os.path.join(tgt_dataset_root, mode, f[:-4]))
            cv2.imwrite(os.path.join(tgt_dataset_root, mode, f[:-4], 'Img.jpg'), x)
            np.savez(os.path.join(tgt_dataset_root, mode, f[:-4], 'bbox-ldmk.npz'), bbox=bbox, ldmk=geom)
            cv2.imwrite(os.path.join(tgt_dataset_root, mode, f[:-4], 'mask.jpg'), shap)

            cnt += 1

        if (n+1) % 1000 == 0:  print('[*] Preproc: already %06d/%06d done; %06d chosen data' % (n+1, len(files), cnt))

    print('[*] Preproc: all images done, total data number: %06d' % (cnt))
    return

def check_theta(tgt_dataset_root, threshold=4., tolerance=1.):
    for mode in ('train', 'valid'):
        print('[*] ==================== %s part ====================' % mode)
        dirs = os.listdir(os.path.join(tgt_dataset_root, mode))
        for d in dirs:
            check = False

            image = cv2.imread(os.path.join(tgt_dataset_root, mode, d, 'Img.jpg'))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 1)
            if len(dets) == 1:
                det = dets[0]
                shape = predictor(image, det)
                p2d = np.asarray([(shape.part(n).x, shape.part(n).y) for n in range(shape.num_parts)], np.float32)               
                theta = get_pose(det, p2d)

                if theta > 0 and theta < threshold + tolerance:
                    check = True
                else:
                    print('[E] %s.jpg (%.1f): head pose far beyond threshold' % (d, theta))
            else:
                print('[E] %s.jpg: can not detect unique face' % d)

            if not check:
                shutil.rmtree(os.path.join(tgt_dataset_root, mode, d))

    print('[*] ==================== check done ====================')

def preproc(opt):
    print('[*] =================== preproc start ==================')
    for mode in ('train', 'valid'):
        print('[*] ==================== %s part ====================' % mode)
        dirs = os.listdir(os.path.join(opt.dataset_root, mode))
        for d in dirs:
            img = cv2.imread(os.path.join(opt.dataset_root, mode, d, 'Img.jpg'))
            mask = cv2.imread(os.path.join(opt.dataset_root, mode, d, 'mask.jpg'))
            p2d = np.load(os.path.join(opt.dataset_root, mode, d, 'bbox-ldmk.npz'))['ldmk']

            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            y = np.zeros_like(x)
            for pt in p2d.astype(np.int):
                cv2.circle(y, tuple(pt), 2, (255,255,255), -1)
            y[:, :, 2] = mask[:, :, 2]

            np.savez(os.path.join(opt.dataset_root, mode, d, 'x-y.npz'), x=x, y=y)
    
    print('[*] =================== preproc done ===================')

class CelebAF_Dataset(Dataset):
    def __init__(self, mode, opt):
        assert(mode == 'train' or mode == 'valid')
        assert(opt.image_size == 256)
        self.mode = mode
        self.image_size = opt.image_size
        
        self.data_path = os.path.join(opt.dataset_root, mode)
        if not os.path.exists(self.data_path):
            gen_dataset(raw_dataset_root='/staff/isc20/vae-gan/landmark-cvae/CelebA/', tgt_dataset_root=opt.dataset_root)
            check_theta(tgt_dataset_root=opt.dataset_root)
            
        self.dirs = os.listdir(self.data_path)
        if not os.path.exists(os.path.join(self.data_path, self.dirs[0], 'x-y.npz')):
            preproc(opt)

        print('[*] CelebA-F dataset (%s mode): totally %d data' % (self.mode, len(self.dirs)))

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
        return len(self.dirs)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_path, self.dirs[idx], 'x-y.npz'))
        x, y = data['x'], data['y']

        # ndarray->tensor & augmentation for training set
        x, y = self.postproc(x, y)
        
        return x, y

# class myDataLoader(object):
#     def __init__(self, 
#                  dataset, 
#                  batch_size=1, 
#                  num_workers=4,
#                  drop_last=False, 
#                  shuffle=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.drop_last = drop_last
#         self.shuffle = shuffle

#     def __len__(self):
#         N = len(self.dataset)
#         return N // self.batch_size if self.drop_last else ((N-1)//self.batch_size + 1)

#     def __iter__(self):
#         # initialize indices and pos
#         self.indices = list(range(len(self.dataset)))
#         if self.shuffle:
#             random.shuffle(self.indices)
#         if self.drop_last and len(self.indices) % self.batch_size != 0:
#             self.indices = self.indices[:-(len(self.indices) % self.batch_size)]
#         self.pos = 0
#         self.queue = Queue()
#         self.fetch_next_batch()    # go to fetch the first batch
#         return self

#     def __next__(self):
#         self.wait_this_batch()

#         if self.queue.empty():
#             raise StopIteration()
#         xs = []
#         ys = []
#         while not self.queue.empty():
#             x, y = self.queue.get()
#             xs.append(x)
#             ys.append(y)
#         X = torch.cat(xs, dim=0)
#         Y = torch.cat(ys, dim=0)
        
#         self.fetch_next_batch()
#         return X, Y

#     def wait_this_batch(self):
#         for t in self.threads:
#             t.join()
#         self.pos = min(self.pos+self.batch_size, len(self.indices))

#     def load_fn(self, begin, end):
#         if begin < end:
#             xs = []
#             ys = []
#             for i in range(begin, end):
#                 x, y = self.dataset[self.indices[i]]
#                 xs.append(x)
#                 ys.append(y)
#             X = torch.stack(xs, dim=0)
#             Y = torch.stack(ys, dim=0)             
#             self.queue.put((X, Y))

#     def fetch_next_batch(self):
#         L = min(self.pos+self.batch_size, len(self.indices)) - self.pos
#         N = self.num_workers

#         self.threads = [Thread(target=self.load_fn, args=(self.pos+L*i//N, self.pos+L*(i+1)//N)) for i in range(N)]
#         for t in self.threads:
#             t.start()

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    import time

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset-root', type=str, default='./CelebA')
    arg_parser.add_argument('--image-size', type=int, default=256)
    opt = arg_parser.parse_args()
    dataset = CelebAF_Dataset('train', opt)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    print(len(dataloader))
    start = time.time()
    for x, y in dataloader:
        assert(x.size(0) == 16 and y.size(0) == 16)
    print('[*] duration: %.1f s' % (time.time()-start))
