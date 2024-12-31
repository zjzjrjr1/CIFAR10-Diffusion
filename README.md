# Project Name: Diffusion Model with CIFAR10

## Project explanation
The project aims to recreate the denoising diffusion model paper with CIFAR10. 
For a complete report and explanation of the code in detail, please visit my Medium article:
https://medium.com/@ikim1994914/fundamental-generative-ai-part-2c-u-net-with-diffusion-model-b53df41b1281

## Code explanation
There are several hyperparameters you can tune. 
1. Beta scheduler : this one uses linear scheduler. Feel free to change the timesteps, min and max beta.
2. Self Attention: Instead of single head attention, I implemented multihead attention. Feel free to change the number of heads.
3. self.ch_list: this is the number of channels of CNN as you go down the list. default is set to [3,64,128,256,512] It starts with 3 for RGB.
4. attention layer: it determines at what layer the attention should be set at. For example, if attention_layer = [1,2], it will use output from the 64 channel and 128 channel output on U-net on both going down and going up the Unet
5. step: this is how many steps of diffusion noise adding you want.
6. temb_dim: this is the original time embedding dimension. In the code, this gets converted into CNN channle dimension.
   For example, t = 57 outputs 256 size vector. But if the CNN output is 512 channels (at the 4th layer of the Unet), then it goes through nn.Linear and produces 512 dimension output to be added to the CNN output.

## Results: 
Bit of a situation here...My desktop is getting shipped from Japan to the US and I will not have it for another 3 months. 
With my laptop, which has MX150 GPU, I cannot do muchh training. Below is what I got after 3 epochs. 
Diffusion model are usually trained for about 100 epochs or more on small dataset such as CIFAR10. 
I will upload better results after I get my better GPU. 
![image](https://github.com/user-attachments/assets/64a887ad-0e11-4d1b-ba74-9cdf63288044)
As you can see, the image blobs started to form, but not a visible object has formed yet. 
