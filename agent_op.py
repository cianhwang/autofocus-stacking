#!/usr/bin/env python
# coding: utf-8


# ## Focus Ctrl Module

# In[9]:


import os
from model import *
import warnings
warnings.simplefilter("ignore", UserWarning)
from quad_solver import *
import torchvision

def load_checkpoint(ckpt_path, model):
    
    ckpt_dir = 'ckpt/'+ckpt_path

    print("[*] Loading model from {}".format(ckpt_dir))

    ckpt_dir = '.'
    filename = 'rfc_model_best.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)

    # load variables from checkpoint
    start_epoch = ckpt['epoch']
    best_loss = ckpt['best_valid_mse']
    print("current epoch: {} --- best loss: {}".format(start_epoch, best_loss))
    model.load_state_dict(ckpt['model_state'])
    #optimizer.load_state_dict(ckpt['optim_state'])   

    return model

def load_rlmodel():
    rlmodel = focusLocNet(0.17, 1, 256, 2).to("cuda:0")
    rlmodel = load_checkpoint('best_model', rlmodel)
    return rl_model


def reset():
    h = [torch.zeros(1, 1, 256).cuda(),
                  torch.zeros(1, 1, 256).cuda()]
    l = torch.rand(1, 2).cuda()*2.0-1.0 #-0.5~0.5
    return h, l

def dist_from_region(n_img, loc):
    
    assert len(n_img.shape) == 2

    H, W = n_img.shape
    window_size =  512

    x_l = int((loc[0]+1) * (H - window_size) / 2)
    y_l = int((loc[1]+1) * (W - window_size) / 2)
    x_r = int(min(H, x_l + window_size))
    y_r = int(min(W, y_l + window_size))

    dist = np.mean(n_img[x_l:x_r, y_l:y_r])

    return dist

    

if __name__ == '__main__':

    with torch.no_grad():

        h, l = reset()
        
        fused_nimg = dist_est(rgb2gray(cv2.resize(prev_img, None, fx = 2, fy = 2)))
        n_img_resized = (cv2.resize(fused_nimg, (128, 64)) /8.0 * 255.0).astype(np.uint8)
        input_t = torch.tensor(n_img_resized/127.5 - 1.0).float().cuda().unsqueeze(0).unsqueeze(0)
        ## RL 128x64
        h, mu, l, b, p = rlmodel(input_t, l, h)
        dist_to_move = -dist_from_region(n_img, mu.squeeze().cpu().numpy())
        
        input_show = torch.tensor(fused_nimg/8.0).float().cuda().unsqueeze(0).unsqueeze(0)
        
        n_img_w_region = torchvision.utils.make_grid(color_region(input_show.repeat(1, 3, 1, 1), mu)).cpu()
        imshow(n_img_w_region)
        cv2.imwrite("rl_frames/nimg{:04d}_{:02d}.png".format(Time, i), (n_img_w_region.numpy().transpose(1, 2, 0)[...,::-1] * 255.0).astype(np.uint8))    
        
        curr = curr + solver(curr, dist_to_move)        
        curr = np.clip(curr, 450, 1000)
        print("next curr: {:.1f}, dist to move: {:.1f}".format(curr, dist_to_move))
 



