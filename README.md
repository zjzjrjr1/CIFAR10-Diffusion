# Project Name: Diffusion Model with CIFAR10

## Project explanation
The project aims to recreate the denoising diffusion model paper with CIFAR10 + MNIST. 
For a complete report and explanation of the code in detail, please visit my Medium article:
https://medium.com/@ikim1994914/fundamental-generative-ai-part-2c-u-net-with-diffusion-model-b53df41b1281

## Code explanation
Switch between greyscale and RGB: 
change "greyscale" to True or False to make plotting work. 
if greyscale, make sure you change segmentation = 1. if RGB, change segmentation = 3. 
If grescale, make channel list to start with 1 . ex. [1,64,128]. If RGB, make channel list to start with 3. 

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
With my laptop, which has MX150 GPU, I cannot do muchh training.
Below is what I got after the first epoch on MNIST set. 
![image](https://github.com/user-attachments/assets/6a297d64-5fd0-4d13-9439-e93caea9b5a6)
This is what I got after 20 epochs. 
![image](https://github.com/user-attachments/assets/666d6591-8b05-4a67-a97f-20d3dbf1f3d3)

As you can see, the images after 20 runs start to make some images that look very close to numbers. 
I will do this for CIFAR10 once I get better images. 
