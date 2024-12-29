import torch.nn as nn
import torch.nn.functional as F
import torch

# this model uses batch normalization bc for cifar32, i can have sufficient amount of batch per GPU
class unet_self_attention(nn.Module):
    def __init__(self, segmentation = 3, temb_dim = 256, dropout = 0.05, 
                 channel_list = [3,64,128,256,512], attention_layer_list = [1,2], num_head = 8):
        super(unet_self_attention, self).__init__()

        self.seg = segmentation
        self.temb_dim = temb_dim 
        self.dropout = dropout
        self.num_head = num_head
        self.ch_list = channel_list # [3,64,128,256,512]
        self.num_layer = len(channel_list) - 1 # 4: top to the 4th, excluding the middle portion     
        self.att_layer_list = attention_layer_list  # at what layer the attention should be applied
        
        # down_cnn1, down_cnn2, up_cnn1, up_cnn2, deconv
        self.down_cnn1 = nn.ModuleList();       self.down_cnn2 = nn.ModuleList()
        self.down_bn1 = nn.ModuleList();        self.down_bn2 = nn.ModuleList()
        self.up_cnn1 = nn.ModuleList();         self.up_cnn2 = nn.ModuleList();
        self.up_bn1 = nn.ModuleList();          self.up_bn2 = nn.ModuleList()
        self.deconv = nn.ModuleList()          
        self.up_res = nn.ModuleList();          self.down_res = nn.ModuleList()
        self.down_attention = nn.ModuleList();  self.up_attention = nn.ModuleList()
        self.down_Wq = nn.ParameterList();      self.down_Wv = nn.ParameterList();  self.down_Wk = nn.ParameterList()
        self.up_Wq = nn.ParameterList();        self.up_Wv = nn.ParameterList();    self.up_Wk = nn.ParameterList()  
        self.down_temb_layer = nn.ModuleList(); self.up_temb_layer = nn.ModuleList()
        for i_l in range(self.num_layer): #i_l  = 0,1,2,3
        
            #### for down cnn, it starts with 3 at i_l = 0, and ends with 512 ati_l = 3 ####
            self.down_cnn1.append(nn.Conv2d(in_channels = self.ch_list[i_l],
                                            out_channels = self.ch_list[i_l + 1],
                                            kernel_size = 3, padding = 1))
            self.down_bn1.append(nn.BatchNorm2d(num_features = self.ch_list[i_l]).to('cuda'))
            
            self.down_cnn2.append(nn.Conv2d(in_channels = self.ch_list[i_l + 1],
                                            out_channels = self.ch_list[i_l + 1],
                                            kernel_size = 3, padding = 1).to('cuda'))
            self.down_bn2.append(nn.GroupNorm(num_groups = 8, num_channels = self.ch_list[i_l + 1]).to('cuda'))
            #self.down_bn2.append(nn.BatchNorm2d(num_features = self.ch_list[i_l + 1]).to('cuda'))
            
            #### for up cnn, it start with 1024, ends with 64 ####
            self.up_cnn1.append(nn.Conv2d(in_channels = 2*self.ch_list[self.num_layer - i_l],
                                          out_channels = self.ch_list[self.num_layer - i_l],
                                          kernel_size = 3, padding = 1).to('cuda'))
            self.up_bn1.append(nn.GroupNorm(num_groups = 8, num_channels = 2*self.ch_list[self.num_layer - i_l]).to('cuda'))
            #self.up_bn1.append(nn.BatchNorm2d(num_features = 2*self.ch_list[self.num_layer - i_l]).to('cuda'))
            
            self.up_cnn2.append(nn.Conv2d(in_channels = self.ch_list[self.num_layer - i_l],
                                          out_channels = self.ch_list[self.num_layer - i_l],
                                          kernel_size = 3, padding = 1).to('cuda'))
            self.up_bn2.append(nn.GroupNorm(num_groups = 8, num_channels = self.ch_list[self.num_layer - i_l]).to('cuda'))
            #self.up_bn2.append(nn.BatchNorm2d(num_features = self.ch_list[self.num_layer - i_l]).to('cuda'))
            
            #### for deconvolution, starts with 1024 -> 512 ends with 128 -> 64 ####
            self.deconv.append(nn.ConvTranspose2d(in_channels = 2*self.ch_list[self.num_layer - i_l],
                                                  out_channels = self.ch_list[self.num_layer - i_l],
                                                  kernel_size = 2, stride = 2).to('cuda'))
            
            #### for resent, should be same as cnn1 ####
            self.down_res.append(nn.Conv2d(in_channels = self.ch_list[i_l],
                                            out_channels = self.ch_list[i_l + 1],
                                            kernel_size = 3, padding = 1).to('cuda'))
            self.up_res.append(nn.Conv2d(in_channels = 2*self.ch_list[self.num_layer - i_l],
                                            out_channels = self.ch_list[self.num_layer - i_l],
                                            kernel_size = 3, padding = 1).to('cuda'))
            
            # create attention when the layer matches
            if i_l in self.att_layer_list:
                down_dim = self.ch_list[i_l + 1] # need + 1, because the attention starts after the cnn and res block
                self.down_attention.append(nn.MultiheadAttention(embed_dim = down_dim, num_heads = self.num_head, batch_first = True).to('cuda'))
                self.down_Wq.append(nn.Parameter(torch.randn(down_dim,down_dim)).to('cuda'))
                self.down_Wv.append(nn.Parameter(torch.randn(down_dim,down_dim)).to('cuda'))
                self.down_Wk.append(nn.Parameter(torch.randn(down_dim,down_dim)).to('cuda'))
                
                up_dim = self.ch_list[self.num_layer - i_l]
                self.up_attention.append(nn.MultiheadAttention(embed_dim = up_dim, num_heads = self.num_head, batch_first = True).to('cuda'))
                self.up_Wq.append(nn.Parameter(torch.randn(up_dim,up_dim)).to('cuda'))
                self.up_Wv.append(nn.Parameter(torch.randn(up_dim,up_dim)).to('cuda'))
                self.up_Wk.append(nn.Parameter(torch.randn(up_dim,up_dim)).to('cuda'))
            # create time embedding FC net. should go from specified input to [64,128,250,512]
            self.down_temb_layer.append(nn.Linear(in_features = self.temb_dim, out_features = self.ch_list[i_l + 1]).to('cuda'))
            self.up_temb_layer.append(nn.Linear(in_features = self.temb_dim, out_features = self.ch_list[self.num_layer - i_l]).to('cuda'))
            
            
        # create the middle layer
        self.middle_cnn1 = nn.Conv2d(in_channels = self.ch_list[-1], out_channels = 2*self.ch_list[-1],
                                     kernel_size=3, padding=1).to('cuda')
        self.middle_res = nn.Conv2d(in_channels = self.ch_list[-1], out_channels = 2*self.ch_list[-1],
                                    kernel_size=3, padding=1).to('cuda')
        self.middle_cnn2 = nn.Conv2d(in_channels = 2*self.ch_list[-1], out_channels = 2*self.ch_list[-1],
                                     kernel_size=3, padding=1).to('cuda')
        self.middle_bn1 = nn.BatchNorm2d(num_features = self.ch_list[-1]).to('cuda')
        self.middle_bn2 = nn.BatchNorm2d(num_features = 2*self.ch_list[-1]).to('cuda')
        self.middle_temb = nn.Linear(in_features = self.temb_dim, out_features = 2*self.ch_list[-1]).to('cuda')
        
        # create the final layer and other misc layers
        self.final_cnn = nn.Conv2d(in_channels = self.ch_list[1], out_channels = self.seg,
                                   kernel_size = 3, padding = 1).to('cuda')
        self.relu = nn.SiLU().to('cuda')
        self.down_sample = nn.AvgPool2d(kernel_size = 2, stride = 2).to('cuda')
        
    
    # create the res_block
    def res_block(self, x, cnn1, cnn2, res_cnn, bn1, bn2, temb):
        temb = temb[:,:,None, None]
        # equation: norm+relu -> cnn1 + temb -> norm + relu -> dropout -> cnn2 + res_cnn(x)
        x_relu_norm = self.relu(bn1(x))
        cnn1_output = cnn1(x_relu_norm)
        # add the time embedding  
        cnn1_temb = cnn1_output + temb
        # normalize and non linear
        cnn2_input = self.relu(bn2(cnn1_temb))
        # add drop out
        cnn2_input = F.dropout(cnn2_input, p = self.dropout)
        down_cnn2_output = cnn2(cnn2_input)
        x = res_cnn(x)
        return x + down_cnn2_output
        
    # create time embedding output
    def time_embed(self, timesteps):
        # equation:
        # PE(pos = 2i) = sin(pos/n^(2i/d))
        # PE(pos = 2i+1) = cos(pos/n^(2i/d))
        n = 10000
        # create the PE tensor. 
        emb = torch.zeros(self.temb_dim).to('cuda')
        i = torch.arange(0, self.temb_dim/2).to('cuda')
        sin_output = torch.sin(timesteps / torch.pow(n, 2*i/self.temb_dim)).to('cuda')
        cos_output = torch.cos(timesteps / torch.pow(n, 2*i/self.temb_dim)).to('cuda')
        emb[::2] = sin_output
        emb[1::2] = cos_output
        return emb # timesteps(1) x self.temb_dim

    # create the forward
    def forward(self, x, t):
        B,_,_,_ = x.shape
        time_embed = self.time_embed(t).detach()
        time_embed = time_embed.repeat(B, 1)
        down_output_list = []
        att_idx = 0
        for i_l, (down_cnn1, down_cnn2, res_cnn, bn1, bn2, temb_fc) in enumerate(zip(self.down_cnn1, self.down_cnn2,
                                                          self.down_res, self.down_bn1, self.down_bn2, self.down_temb_layer)):
            temb = temb_fc(time_embed)  # B x time_embed -> B x feature dim
            res_output = self.res_block(x, down_cnn1, down_cnn2, res_cnn, bn1, bn2, temb)
            
            # run attention
            
            if i_l in self.att_layer_list:
                B, C, H, W = res_output.shape; res_output = torch.permute(res_output, (0,2,3,1))
                res_output_flat = res_output.view(B, H*W, C)
                query = torch.matmul(res_output_flat, self.down_Wq[att_idx])
                value = torch.matmul(res_output_flat, self.down_Wv[att_idx])
                key = torch.matmul(res_output_flat, self.down_Wk[att_idx])
                down_attention = self.down_attention[att_idx]
                att_idx += 1
                attention_out,_ = down_attention(query, value, key)
                attention_out = attention_out.reshape(B,H,W,C)
                attention_out = torch.permute(attention_out, (0,3,1,2))
                down_output_list.append(attention_out)
                x = self.down_sample(attention_out)
            else:
                down_output_list.append(res_output)
                x = self.down_sample(res_output)
        
        # middle unet
        middle_res_output = self.middle_res(x)
        x = self.relu(self.middle_bn1(x))
        middle_temb = self.middle_temb(time_embed)
        middle_cnn1_output = self.middle_cnn1(x)
        middle_cnn1_temb = middle_temb[:,:,None,None] + middle_cnn1_output
        middle_cnn2_input = self.relu(self.middle_bn2(middle_cnn1_temb))
        middle_cnn2_output = self.middle_cnn2(middle_cnn2_input)
        x = middle_res_output + middle_cnn2_output
        
        # up unet
        att_idx = 0
        for i_l, (deconv, up_cnn1, up_cnn2, up_res, bn1, bn2, temb_fc) in enumerate(zip(self.deconv, self.up_cnn1,
                                                                              self.up_cnn2, self.up_res,
                                                                              self.up_bn1, self.up_bn2,
                                                                              self.up_temb_layer)): 
            # do deconv then concate with the output from the down convolution
            deconv_output = deconv(x)
            concat_deconv = torch.concat((deconv_output, down_output_list[self.num_layer - (i_l+1)]), dim = 1)
            temb_fc = self.up_temb_layer[i_l]
            temb = temb_fc(time_embed)
            res_cnn = self.up_res[i_l]
            # attention
            res_output = self.res_block(concat_deconv, up_cnn1, up_cnn2, res_cnn, bn1, bn2, temb)
            
            # run the attention - note that when the up attention was made, the dimension can 
            # just go in order - no need to switch.  
            if i_l in self.att_layer_list:
                B, C, H, W = res_output.shape; res_output = torch.permute(res_output, (0,2,3,1))
                res_output_flat = res_output.view(B, H*W, C)
                query = torch.matmul(res_output_flat, self.up_Wq[att_idx])
                value = torch.matmul(res_output_flat, self.up_Wv[att_idx])
                key = torch.matmul(res_output_flat, self.up_Wk[att_idx])
                up_attention = self.up_attention[att_idx]
                att_idx += 1
                attention_out,_ = up_attention(query, value, key)
                attention_out = attention_out.reshape(B,H,W,C)
                attention_out = torch.permute(attention_out, (0,3,1,2))               
                x = attention_out
            else:
                x = res_output
                # final outpout
        final_out = self.final_cnn(x)
        return final_out