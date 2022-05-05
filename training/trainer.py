import sys
sys.path.append(".")
sys.path.append("..")
import os
from utils.latent_utils import get_latent
from PIL import Image,ImageDraw
from datasets.img_dataset import load_data
import torch
import yaml
from easydict import EasyDict
from utils.utils import tensor2im
from argparse import ArgumentParser
from torch.autograd import Variable
from torch import nn
import numpy as np
from training.train_model import Train_Model
from models.latent2latent import Latent2Latent
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
writer = SummaryWriter()

def init_config(args):
    with open("../configs.yaml",encoding='utf8')as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    configs = EasyDict(configs)
    configs.attribute = args.attribute
    return configs

l1_loss_fn = nn.L1Loss()
l2_loss_fn = nn.MSELoss()

def train(args):
    device = "cuda" if args.gpu else "cpu"
    configs = init_config(args)

    #models
    training_models = Train_Model(configs,device)
    latent_model = Latent2Latent().cuda()
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    training_dataloader,eval_dataloader = load_data(configs.data_path,batch_size=args.batch_size)
    optimizer = torch.optim.Adam(latent_model.parameters(), lr=0.001)
    best_model_loss = None
    
    for epoch in range(1,args.epoch+1):
        train_loss_sum = 0
        eval_loss_sum = 0
        os.makedirs("{}/training_imgs/epoch-{}".format(args.exp_dir,epoch),exist_ok=True)
        print(f"Training - {epoch} begining!")
        for i,img in tqdm(enumerate(training_dataloader),total=len(training_dataloader)):
            latent_model.train()
            save_path = "{}/training_imgs/epoch-{}/{}".format(args.exp_dir,epoch,i)
            loss_cur = train_imgs_batch(img,training_models,latent_model,mtcnn,resnet,configs,save_path,train=True,i=i,device=device)
            train_loss_sum += loss_cur.item()
            optimizer.zero_grad()
            loss_cur.backward()
            optimizer.step()
            if i%100 == 0:
                print("[epoch-{} | i-{} | training loss : {}]".format(epoch,i,train_loss_sum/(i+1)))
            if i%1000==0:
                torch.save(latent_model.state_dict(),'{}/latent_model-{}.pt'.format(args.exp_dir,epoch))
        writer.add_scalar("Loss/train", train_loss_sum/len(training_dataloader),epoch)
        print(f"Eval - {epoch} begining!")
        for _,img in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader)):
            with torch.no_grad():
                latent_model.eval()
                loss_cur = train_imgs_batch(img,training_models,latent_model,mtcnn,resnet,configs,train=False,device=device)
                eval_loss_sum += loss_cur.item()
        if best_model_loss is None or eval_loss_sum<best_model_loss:
            b_loss = "None" if best_model_loss is None else best_model_loss
            print("[best model loss:{}|cur loss:{}]".format(b_loss,eval_loss_sum))
            best_model_loss = eval_loss_sum
            print("epoch-{} get lower loss!Save it now!".format(epoch))
            torch.save(latent_model.state_dict(),'{}/latent_model.pt'.format(args.exp_dir))
    writer.flush() 
    writer.close()

def train_imgs_batch(imgs,t_model,latent_model,mtcnn,resnet,configs,save_path=None,train=True,i=0,device="cpu"):
    with torch.no_grad():
        _,w_latents = get_latent(imgs,t_model.encoder,configs)

    w_a_latents = t_model.latent_editor.apply_interfacegan(w_latents,configs.attribute) # add original attributes to latent code w
    
    w_n = latent_model(w_a_latents)
    w_n = 0.7*w_latents + 0.3*w_n

    #cycle loss
    w_n_c = t_model.latent_editor.apply_interfacegan(w_n,configs.attribute,factor=-configs.directions_factor)
    w_n_c = latent_model(w_n_c)
    w_n_c = 0.7*w_latents+0.3*w_n_c
    loss_cycle = l1_loss_fn(w_n_c,w_latents)
    
    #Identity loss
    w_i = t_model.latent_editor.apply_interfacegan(w_latents,configs.attribute,factor=0)
    w_i = latent_model(w_i)
    w_i = 0.7*w_latents + 0.3*w_i
    loss_identity = l1_loss_fn(w_i,w_latents)

    #neighborhood loss
    loss_neighborhood = l2_loss_fn(w_n,w_latents)

    #attributes loss
    a_images = None
    g_a_images = None
    ori_images = None
    for w in w_a_latents:
        a_image, _ = t_model.generator([w], randomize_noise=False, input_is_latent=True)
        a_images = a_image if a_images is None else torch.cat((a_images,a_image))
    for w in w_n:
        g_a_image, _ = t_model.generator([w], randomize_noise=False, input_is_latent=True)
        g_a_images = g_a_image if g_a_images is None else torch.cat((g_a_images,g_a_image))
    for w in w_latents:
        ori_image,_ = t_model.generator([w], randomize_noise=False, input_is_latent=True)
        ori_images = ori_image if ori_images is None else torch.cat((ori_images,ori_image))
    a_images_01 = a_images*0.5+0.5
    g_a_images_01 = g_a_images*0.5+0.5
    a_score = torch.sigmoid(t_model.classifier(a_images_01))
    g_a_score = torch.sigmoid(t_model.classifier(g_a_images_01))
    loss_attributes = l2_loss_fn(g_a_score,a_score)
    
    #Fid loss
    ori_faces_embedding = None
    g_a_faces_embedding = None
    for idx in range(len(ori_images)):
        ori_image = tensor2im(ori_images[idx])
        g_a_image = tensor2im(g_a_images[idx])
        ori_faces = mtcnn(ori_image)
        g_a_faces = mtcnn(g_a_image)
        if ori_faces is None or g_a_faces is None:
            ori_face_embedding, g_a_face_embedding = torch.zeros((1,512)).to(device) , torch.zeros((1,512)).to(device)
        else:
            ori_face_embedding = resnet(ori_faces[0].unsqueeze(0).to('cuda'))
            g_a_face_embedding = resnet(g_a_faces[0].unsqueeze(0).to('cuda'))
        ori_faces_embedding = ori_face_embedding if ori_faces_embedding is None else torch.cat((ori_faces_embedding,ori_face_embedding))
        g_a_faces_embedding = g_a_face_embedding if g_a_faces_embedding is None else torch.cat((g_a_faces_embedding,g_a_face_embedding))
    loss_fid = l1_loss_fn(g_a_faces_embedding,ori_faces_embedding)

    loss_cur = loss_cycle*0.5+loss_attributes+loss_identity*0.5+loss_neighborhood*0.5+loss_fid

    if i%100==0 and train:
        for imgid in range(len(g_a_images)):
            g_img = np.array(tensor2im(g_a_images[imgid]))
            o_img =  np.array(tensor2im(ori_images[imgid]))
            img = np.concatenate((o_img,g_img),axis=1)
            img = Image.fromarray(img)
            img.save("{}_{}.png".format(save_path,imgid))
    return loss_cur

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--exp_dir', type=str,default=None,help='Path to experiment output directory')
    parser.add_argument('--attribute', type=str,default=None,help='attribute for face-edit training')
    parser.add_argument('--gpu', type=bool, default=None)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    train(args)
