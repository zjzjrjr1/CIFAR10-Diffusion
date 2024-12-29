# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 06:52:47 2024

@author: ra064640
"""
from unet_self_attention_pt import unet_self_attention
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random 
import gc

batch = 64
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1, 1, 1])  # Normalize RGB channels
])


# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

# DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch, shuffle=False)   
# create diffusion model for training. 
# create pre defined scheduler. 
# randomly sample t-1 from the scehduler. 
# create the t-1 image, add noise epsilon, and get xt. 

# we have epsilon, t, xt and x0. train beghins. 

def train_epoch(model, loss, optimizer, beta_vec, alpha_vec, step, train_dataloader):
    
    for train_idx, (features, labels) in enumerate(train_dataloader):
        batch, _,_,_ = features.shape
        ## eps are separate for each RGB, because sampled independently of the channel
        eps = torch.randn_like(features)
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            beta_vec = beta_vec.cuda()
            alpha_vec = alpha_vec.cuda()
            eps = eps.cuda()
        ## randint is inclusive at low, exclusive at high
        ## to sample from alpha_vec with t, make sure t is from 0(inclusive) to step (exclusive)
        t = random.randint(0, step)
        ## x0 -> alpha0 -> x1 -> alpha_1 -> x2 -> alpha_2 -> xT: 3 steps. 
        ## if sample until alpha[t], then alpha[t-1] is included
        ## alpha[t-1] is used to calculate xt
        alpha_t_his = alpha_vec[0:(t+1)]
        alpha_t_his_log = torch.log(alpha_t_his)
        alpha_bar_t = torch.sum(alpha_t_his_log).exp()
        xt = torch.sqrt(alpha_bar_t)*features + torch.sqrt(1-alpha_bar_t)*eps
        optimizer.zero_grad()
        
        ## given xt, and t, the model has to guess what noise has been added 
        ## eps is the real noise. 
        eps_pred = model(xt, t)
        eps_flat = torch.flatten(eps)
        eps_pred_flat = torch.flatten(eps_pred)
        output = loss(eps_flat, eps_pred_flat)
        if train_idx % 10 == 0:
            print(f"batch loss is: {torch.sqrt(output).detach()}")
        # backpropagatin
        output.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return features.shape
        
# start the training 
step = 100
# segmentation should be 3 for RGB value 
model = unet_self_attention(segmentation = 3, temb_dim = 256, dropout = 0.05, 
             channel_list = [3,64,128,256], attention_layer_list = [2])

model.train()
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00015)
epoches = 200
loss = nn.MSELoss()
beta_min = 1e-4
beta_max = 0.02
## linspace is min and max inclusive, tensor of size step
beta_vec = torch.linspace(beta_min, beta_max, step) 
alpha_vec = 1 - beta_vec
for epoch in range(epoches):
    feature_size = train_epoch(model, loss, optimizer, beta_vec, alpha_vec, step, train_loader)

    if epoch % 3 == 0 and epoch > 1:
        # start the evaluation loop- unconoditonal generation
        model.eval()
        sample_num = 3
        for j in range(sample_num):
            ## initialize: note that feature_size[0] is batch, so start from 1
            xt = torch.randn(size=(1, feature_size[1], feature_size[2], feature_size[3]))
            # move to cuda
            if torch.cuda.is_available():
                xt  = xt.cuda()
            for i in range(step):
                ## since step = 1000 when i = 999, then 999 < (1000-1) is false, 
                ## so go to the else statement for the last step
                if i < (step-1):
                    ## +1 just to make index out of bounds error correct. 
                    alpha_t = alpha_vec[step - (i+1)]
                    alpha_t_his = alpha_vec[0:(step - (i))]
                    alpha_bar = torch.sum(torch.log(alpha_t_his).exp())
                    beta_t = beta_vec[step - (i+1)]
                    sig_t = torch.sqrt(beta_t)
                    ## move to cuda
                    if torch.cuda.is_available():
                        alpha_t = alpha_t.cuda().detach()
                        beta_t = beta_t.cuda().detach()
                        sig_t = sig_t.cuda().detach()
                    ## estimate the noise 
                    ## t = random.randint(0, (step-1))
                    t = random.randint(0, (step-1))
                    with torch.no_grad():
                        noise = model(xt, t).detach()
                    xt = 1/torch.sqrt(alpha_t)*(xt - (beta_t)/torch.sqrt(1-alpha_bar)*noise) + \
                        sig_t*(torch.randn(size=(1,feature_size[1], feature_size[2], feature_size[3]))).cuda().detach()
                else:
                    xt = 1/torch.sqrt(alpha_t)*(xt - (1-alpha_t)/torch.sqrt(1-alpha_t)*noise)
                if i % 200 == 0:
                    print(i)   
            # display the image of xt
            # Normalize to [0, 1]
            min_val = xt.min()
            max_val = xt.max()
            image_tensor_normalized = (xt - min_val) / (max_val - min_val)
            
            # Scale to [0, 255] and convert to uint8
            image_tensor_scaled = (image_tensor_normalized * 255).clamp(0, 255).byte()
            
            # Convert to NumPy array
            image_np = image_tensor_scaled.cpu().squeeze().permute(1, 2, 0).numpy()
            
            # Display using Matplotlib
            plt.imshow(image_np)
            title = str(epoch) + ' ' + str(i)
            plt.title(title)
            plt.axis('off')
            plt.show()
        model.train()
    
        
    
                                                
        