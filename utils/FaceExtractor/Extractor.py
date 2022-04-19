# Face alignment demo
# Uses MTCNN or FaceBoxes as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Cunjian Chen (ccunjian@gmail.com), Aug 2020

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import scipy.ndimage
from torchvision.transforms.functional import to_tensor
from .common.utils import BBox
from .models.basenet import MobileNet_GDConv
from .models.mobilefacenet import MobileFaceNet
from .Retinaface import Retinaface

map_location='cpu'

def rot90(v):
    return np.array([-v[1], v[0]])

class Extractor(nn.Module):
    def __init__(self, backbone='MobileNet'):
        super().__init__()
        thisdir = Path(__file__).parent
        if backbone == 'MobileNet':
            self.keypoint_model = MobileNet_GDConv(136)
            checkpoint = torch.load(thisdir / 'checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar', map_location=map_location)
            checkpoint = self.rename_keys(checkpoint)
        elif backbone == 'MobileFaceNet':
            self.keypoint_model = MobileFaceNet([112, 112], 136)
            checkpoint = torch.load(thisdir / 'checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)
        else:
            raise ValueError('Wrong backbone specified')

        self.keypoint_model.load_state_dict(checkpoint['state_dict'])
        self.keypoint_model = self.keypoint_model.eval()
        self.out_size = 224  # highest resolution available for MobileNet

        self.detector = Retinaface.Retinaface()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.eye_left_idxs = np.arange(36, 42)
        self.eye_right_idxs = np.arange(42, 48)

    def rename_keys(self, checkpoint):
        new_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        checkpoint['state_dict'] = new_state_dict
        return checkpoint

    def get_landmarks(self, cropped_face):
        if cropped_face.shape[0] <= 0 or cropped_face.shape[1]<=0:
            raise ValueError('Something went wrong')

        cropped_face = (cropped_face - self.mean) / self.std
        landmark = self.keypoint_model(cropped_face)
        landmark = landmark.reshape(-1, 2)
        return landmark

    def retinanet_predict(self, image_01):
        faces = self.detector(image_01)
        if len(faces) == 0:
            raise RuntimeError('Could not detect face in the image')
        if len(faces) > 1:
            sizes = []
            for face in faces:
                x1, y1, x2, y2 = face[:4]
                size = (x2 - x1) * (y2 - y1)
                sizes.append(size)
            max_size_idx = sizes.index(max(sizes))
            face = faces[max_size_idx]
        else:
            face = faces[0]
        return face

    def get_face_bbox(self, image_01):
        height, width = self.get_output_image_size(image_01)

        face = self.retinanet_predict(image_01)
        face = [round(x) for x in face]

        x1=face[0]
        y1=face[1]
        x2=face[2]
        y2=face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h])*1.2)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)

        cropped_face = image_01[:, :, new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            pad_f = torch.nn.ZeroPad2d(padding=(dx, edx, dy, edy))
            cropped_face = pad_f(cropped_face)
        return new_bbox, cropped_face


    def get_output_image_size(self, image_01):
        in_hw = torch.tensor(image_01.shape[2:]).float()
        resize_ratio = 640 / max(in_hw)
        out_hw = resize_ratio * in_hw
        out_hw = torch.round(out_hw)

        height, width = out_hw.int()
        return height.item(), width.item()


    def celebahq_style_extraction(self, img, lm):
        if not isinstance(lm, np.ndarray):
            lm = lm.numpy()

        eye_avg = np.mean((lm[self.eye_left_idxs] + lm[self.eye_right_idxs]) * 0.5 + 0.5, axis=0)
        mouth_avg = (lm[48] + lm[54]) * 0.5 + 0.5
        eye_to_eye = np.mean(lm[self.eye_right_idxs] - lm[self.eye_left_idxs], axis=0)

        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = 1024 / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(1024 * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Simulate super-resolution.
        superres = int(np.exp2(np.ceil(np.log2(zoom))))
        if superres > 1:
            img = img.resize((img.size[0] * superres, img.size[1] * superres), Image.ANTIALIAS)
            quad *= superres
            zoom /= superres

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.mgrid[:h, :w, :1]
            mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(y) / pad[1]), np.minimum(np.float32(w-1-x) / pad[2], np.float32(h-1-y) / pad[3]))
            blur = 1024 * 0.02 / zoom
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
            quad += pad[0:2]

        # Transform.
        img = img.transform((4096, 4096), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        img = img.resize((1024, 1024), Image.ANTIALIAS)
        return img

    # Crop.
    def celebahq_style_reverse(self, img, lm,cropimg):
        
        if not isinstance(lm, np.ndarray):
            lm = lm.numpy()

        eye_avg = np.mean((lm[self.eye_left_idxs] + lm[self.eye_right_idxs]) * 0.5 + 0.5, axis=0)
        mouth_avg = (lm[48] + lm[54]) * 0.5 + 0.5
        eye_to_eye = np.mean(lm[self.eye_right_idxs] - lm[self.eye_left_idxs], axis=0)

        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = 1024 / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(1024 * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            h,w= crop[3] - crop[1],crop[2] - crop[0]
            cropimg = F.interpolate(cropimg, (h, w), mode='area')
            img = to_tensor(img)
            img[:,crop[1]:crop[3],crop[0]:crop[2]] = cropimg.squeeze(0)
        return img

    #paste to origin image
    def reverse_align(self,image_01,crop_image):
        if image_01.ndim == 3:
            image_01 = image_01.unsqueeze(0)
        if crop_image.ndim == 3:
            crop_image = crop_image.unsqueeze(0)
        assert image_01.size(0) == 1, 'The model only accepts a single image'

        height, width = self.get_output_image_size(image_01)
        image_01 = F.interpolate(image_01, (height, width), mode='area')

        new_bbox, cropped_face = self.get_face_bbox(image_01)
        # cropped_face = F.interpolate(cropped_face, (self.out_size, self.out_size), mode='area')  # 224 x 224
        
        landmark = self.get_landmarks(cropped_face)
        landmark = new_bbox.reprojectLandmark(landmark.cpu())
        resize_ratio = 640 / max(torch.tensor(image_01.shape[2:]).float())
        landmark /= resize_ratio

        img_pil = image_01.squeeze().permute(1,2,0).cpu().numpy()
        img_pil = Image.fromarray(np.uint8(img_pil * 255))
        img_tensor = self.celebahq_style_reverse(img_pil, landmark,crop_image)

        return img_tensor

    def __call__(self, image_01):  # [B x 3 x H x W]
        if image_01.ndim == 3:
            image_01 = image_01.unsqueeze(0)
        assert image_01.size(0) == 1, 'The model only accepts a single image'

        height, width = self.get_output_image_size(image_01)
        image_01 = F.interpolate(image_01, (height, width), mode='area')

        new_bbox, cropped_face = self.get_face_bbox(image_01)
        cropped_face = F.interpolate(cropped_face, (self.out_size, self.out_size), mode='area')  # 224 x 224
        
        landmark = self.get_landmarks(cropped_face)
        landmark = new_bbox.reprojectLandmark(landmark.cpu())
        resize_ratio = 640 / max(torch.tensor(image_01.shape[2:]).float())
        landmark /= resize_ratio

        img_pil = image_01.squeeze().permute(1,2,0).cpu().numpy()
        img_pil = Image.fromarray(np.uint8(img_pil * 255))
        img_pil = self.celebahq_style_extraction(img_pil, landmark)
        img_tensor = TF.to_tensor(img_pil)

        return img_tensor
