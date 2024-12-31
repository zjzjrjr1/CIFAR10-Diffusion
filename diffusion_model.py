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
import time 
import gc

batch = 128



'''
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.0], std=[1])  # Convert images to tensor
     # Normalize with mean 0.5 and std 0.5 for single channel (MNIST is grayscale)
])
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)  


'''

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
greyscale = False
# create diffusion model for training. 
# create pre defined scheduler. 
# randomly sample t-1 from the scehduler. 
# create the t-1 image, add noise epsilon, and get xt. 

# we have epsilon, t, xt and x0. train beghins. 

def train_epoch(model, loss, optimizer, alpha_vec_expand, step, train_dataloader):
    divider = 10
    for train_idx, (features, labels) in enumerate(train_dataloader):
        start = time.time()
        batch, _,_,_ = features.shape
        ## eps are separate for each RGB, because sampled independently of the channel
        eps = torch.randn_like(features)
        ## randint is inclusive at low, exclusive at high
        ## to sample from alpha_vec with t, make sure t is from 0(inclusive) to step (exclusive)
        t = torch.randint(0, step, (batch,))
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            alpha_vec_expand = alpha_vec_expand.cuda()
            eps = eps.cuda()
            t = t.cuda()
        ## x0 -> alpha0 -> x1 -> alpha_1 -> x2 -> alpha_2 -> xT: 3 steps. 
        ## then t is either 0,1,2
        ## if t = 1, then alpha_1 should be chosen to generate x2
        '''
        alpha_t_his = alpha_vec_expand[0:t]
        alpha_t_his_log = torch.log(alpha_t_his)
        alpha_bar_t = torch.sum(alpha_t_his_log).exp()
        '''
        alpha_vec_log = torch.log(alpha_vec_expand)
        alpha_vec_log_cumsum = torch.cumsum(alpha_vec_log, dim = 1)
        # I need to include t + 1 because [0:t] excludes t. 
        # example, if t = 2 (last step), then need to include alpha_1 and produce x3/xT
        # but it excludes alpha_2. so t+1 to include alpha_2
        alpha_bar_t_log_cumsum = alpha_vec_log_cumsum[torch.arange(batch),t]
        alpha_bar_t = alpha_bar_t_log_cumsum.exp() 
        # when xt is produced, note that it is actual x(t+1) wrt to t.
        # remember, if there are T steps, then there are T+1 xt's. 
        xt = torch.sqrt(alpha_bar_t)[:,None, None, None]*features + torch.sqrt(1-alpha_bar_t)[:,None, None, None]*eps
        optimizer.zero_grad()
        
        ## given xt, and t, the model has to guess what noise has been added 
        ## eps is the real noise. 
        eps_pred = model(xt, t)
        eps_flat = torch.flatten(eps)
        eps_pred_flat = torch.flatten(eps_pred)
        output = loss(eps_flat, eps_pred_flat)
        
        output.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        end = time.time()
        
        if train_idx % divider == 0:
            print(f"batch loss is: {torch.sqrt(output).detach()} at {train_idx} took {(end-start)*divider}")
        # backpropagatin
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return features.shape
        
# start the training 
step = 500
# segmentation should be 3 for RGB value 
model = unet_self_attention(segmentation = 3, temb_dim = 256, dropout = 0.00, 
             channel_list = [3,64,128,256, 512], attention_layer_list = [1,2])

model.train()
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
epoches = 100
loss = nn.MSELoss()
beta_min = 10e-4
beta_max = 0.03

## linspace is min and max inclusive, tensor of size step
beta_vec = torch.linspace(beta_min, beta_max, step) 
alpha_vec = 1 - beta_vec
alpha_vec_expand = alpha_vec.expand(batch, -1)
for epoch in range(epoches):
    feature_size = train_epoch(model, loss, optimizer, alpha_vec_expand, step, train_loader)

    if epoch % 2 == 0:
        # start the evaluation loop- unconoditonal generation
        model.eval()
        sample_num = 5
        for j in range(sample_num):
            ## initialize: note that feature_size[0] is batch, so start from 1
            xt = torch.randn(size=(1, feature_size[1], feature_size[2], feature_size[3]))
            # move to cuda
            if torch.cuda.is_available():
                xt  = xt.cuda()
            for i in reversed(range(step)):
                ## since step = T, there must be T+1 images produced
                # x0 is already produced at the initialization, so only T images need to be produced 
                # in this loop. 
                if i != 0:
                    ##  need t+1 to include the t. 
                    alpha_t = alpha_vec[i]
                    alpha_t_his = alpha_vec[0:(i+1)]
                    alpha_bar = torch.sum(torch.log(alpha_t_his)).exp()
                    beta_t = beta_vec[i]
                    sig_t = torch.sqrt(beta_t)
                    ## move to cuda
                    if torch.cuda.is_available():
                        alpha_t = alpha_t.cuda().detach()
                        beta_t = beta_t.cuda().detach()
                        sig_t = sig_t.cuda().detach()
                    ## estimate the noise 
                    with torch.no_grad():
                        noise = model(xt, torch.tensor([i]).cuda()).detach()
                    xt = 1/torch.sqrt(alpha_t)*(xt - (beta_t)/torch.sqrt(1-alpha_bar)*noise) + \
                        sig_t*(torch.randn(size=(1,feature_size[1], feature_size[2], feature_size[3]))).cuda().detach()
                else:
                    xt = 1/torch.sqrt(alpha_t)*(xt - (1-alpha_t)/torch.sqrt(1-alpha_t)*noise) 
                    
                # display the image of xt
                # Normalize to [0, 1]
                if i % 50 == 0:
                    min_val = xt.min()
                    max_val = xt.max()
                    image_tensor_normalized = (xt - min_val) / (max_val - min_val)
                    
                    # Scale to [0, 255] and convert to uint8
                    image_tensor_scaled = (image_tensor_normalized * 255).clamp(0, 255).byte()
                    
                    # Convert to NumPy array
                    if greyscale == True:
                        image_np = image_tensor_scaled.cpu().squeeze().numpy()
                    else:
                        image_np = image_tensor_scaled.cpu().squeeze().permute(1, 2, 0).numpy()
                    
                    # Display using Matplotlib
                    plt.imshow(image_np)
                    title = str(epoch) + '_' + str(i) + ' ' + str(j)
                    plt.title(title)
                    plt.axis('off')
                    plt.show()
            model.train()
    
        
    
                                                
        