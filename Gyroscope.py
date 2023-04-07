import os
import cv2
import datetime
import numpy as np

import time
import argparse
import numpy as np

import torch

import HoLiNet
import Equi2Equi

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=8,suppress=True)

def count_params(a):
    out = sum(p.numel() for p in a.parameters())
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--root_dir', default='Datasets/Disco/Equirect/',
                    help='Test images and labels root directory')
    parser.add_argument('--pth', default='ckpt/HoliNet_weights.pth',
                    help='Weights to use in the network')
    parser.add_argument('--out_dir', default='Results_SVMIS',
                    help='Test images and labels root directory')
    parser.add_argument('--no_cuda', action='store_true',
                    help='Disable cuda use')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_cuda else 'cuda') 

    # Path to 360 images
    img_dir = os.path.join(args.root_dir)
    img_list = os.listdir(img_dir)
    img_list.sort(key=lambda x:int(x.split('.')[0][2:]))
    os.makedirs(args.out_dir,exist_ok=True)
    
    start_t = time.time()
    # Load trained network
    net = HoLiNet.load_weigths(args)
    W,H = 128,64 
    print('Network loaded')

    net.to(device)
    net.eval()
    G_dir = np.array([[0,1,0]]).reshape(1,3)
    out_rot = np.zeros((len(img_list),6))

    for i,name in enumerate(img_list):
        # Read image and rotate upside-down
        img = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(img_dir,name)),(W*2,H*2),cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)

        # Transform to Pytorch tensor and inference
        x = torch.FloatTensor(img.transpose([2,0,1])/255.0).unsqueeze(0)
        with torch.no_grad():
            output = net(x.to(device)).detach().cpu().numpy().squeeze(0)
        VP_pred = output[0]
        HL_pred = output[1]

        # Compute the orientation vector from the output maps of the network
        Plane = Equi2Equi.HL2Plane(HL_pred,VP_pred)
        aaa = np.cross(G_dir,Plane).reshape(-1,)

        # Avoid singularity when the prediction set the image as gravity oriented
        if np.linalg.norm(aaa) != 0:
            aaa = aaa / np.linalg.norm(aaa)
            bbb = np.arccos(np.dot(G_dir,Plane))
        else:
            aaa = np.array([0,0,1])
            bbb = 0.
        out_rot[i,3:] = aaa * bbb

        # Save 1 frame each 500
        if i%500 == 0:
            out_img = np.zeros_like(img)
            out_img[...,0] = cv2.resize((VP_pred/VP_pred.max())*255,(W*2,H*2))
            out_img[...,2] = cv2.resize((HL_pred/HL_pred.max())*255,(W*2,H*2))
            out_img = cv2.addWeighted(img,1.,out_img,1.,0)
            cv2.imwrite(args.out_dir+'/'+name,out_img)


    # Save the results into a txt file
    np.savetxt('{}_HoliNet_2DoF.txt'.format(datetime.date.today()),out_rot)
    end_t = time.time()
    accum_time = (end_t-start_t)

    print('Total time: %.2f s' %accum_time)
    print('Average time per image: %.2f ms' %(1000*accum_time/len(img_list)))
