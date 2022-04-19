import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from utils.FaceExtractor import Extractor
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dirs):
    images = []
    assert os.path.isdir(dirs), '%s is not a valid directory' % dirs

    for root, _, fnames in sorted(os.walk(dirs)):
        fnames.sort()
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
extractor = Extractor.Extractor().to(device)



def crop_reverse(imgpath,cimgpath,savepath):
    imgpic = Image.open(imgpath).convert('RGB')
    cimgpic = Image.open(cimgpath).convert('RGB')
    os.makedirs(os.path.dirname(savepath),exist_ok=True)
    imgpic = to_tensor(imgpic).to(device)
    cimgpic = to_tensor(cimgpic).to(device)
    img_new = extractor.reverse_align(imgpic,cimgpic)
    save_image(img_new.cpu(),savepath)


def crop_align_dir(imgdir,savedir):
    os.makedirs(savedir,exist_ok=True)
    imglist = make_dataset(imgdir)
    for img in tqdm(imglist):
        name = img.split("/")[-1]
        imgpic = Image.open(img).convert('RGB')
        imgpic = to_tensor(imgpic).to(device)
        img_new = extractor(imgpic)
        img_save = os.path.join(savedir,name)
        save_image(img_new.cpu(),img_save)

def crop_align(img,savedir):
    os.makedirs(savedir,exist_ok=True)
    name = img.split("/")[-1]
    imgpic = Image.open(img).convert('RGB')
    imgpic = to_tensor(imgpic).to(device)
    img_new = extractor(imgpic)
    img_save = os.path.join(savedir,name)
    save_image(img_new.cpu(),img_save)
    return img_save



