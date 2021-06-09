#############################
# Feature Visualization
#############################
import torch
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torch import optim
import sys
import cv2
from torchvision import transforms
# Height = 28
# Width = 28
# layer_name='4a'



activation = {} # dictionary to store the activation of a layer
def create_hook(name):
    def hook(m, i, o):
# copy the output of the given layer
        activation[name] = o

    return hook


# class to compute image gradients in pytorch
class RGBgradients(nn.Module):
    def __init__(self, weight): # weight is a numpy array
        super().__init__()
        k_height, k_width = weight.shape[1:]
        # assuming that the height and width of the kernel are always odd numbers
        padding_x = int((k_height-1)/2)
        padding_y = int((k_width-1)/2)
        
        # convolutional layer with 3 in_channels and 6 out_channels 
        # the 3 in_channels are the color channels of the image
        # for each in_channel we have 2 out_channels corresponding to the x and the y gradients
        self.conv = nn.Conv2d(3, 6, (k_height, k_width), bias = False, 
                              padding = (padding_x, padding_y) )
        # initialize the weights of the convolutional layer to be the one provided
        # the weights correspond to the x/y filter for the channel in question and zeros for other channels
        weight1x = np.array([weight[0], 
                             np.zeros((k_height, k_width)), 
                             np.zeros((k_height, k_width))]) # x-derivative for 1st in_channel
        
        weight1y = np.array([weight[1], 
                             np.zeros((k_height, k_width)), 
                             np.zeros((k_height, k_width))]) # y-derivative for 1st in_channel
        
        weight2x = np.array([np.zeros((k_height, k_width)),
                             weight[0],
                             np.zeros((k_height, k_width))]) # x-derivative for 2nd in_channel
        
        weight2y = np.array([np.zeros((k_height, k_width)), 
                             weight[1],
                             np.zeros((k_height, k_width))]) # y-derivative for 2nd in_channel
        
        
        weight3x = np.array([np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width)),
                             weight[0]]) # x-derivative for 3rd in_channel
        
        weight3y = np.array([np.zeros((k_height, k_width)),
                             np.zeros((k_height, k_width)), 
                             weight[1]]) # y-derivative for 3rd in_channel
        
        weight_final = torch.from_numpy(np.array([weight1x, weight1y,weight2x, weight2y,weight3x, weight3y])).type(torch.FloatTensor)
        
        if self.conv.weight.shape == weight_final.shape:
            self.conv.weight = nn.Parameter(weight_final)
            self.conv.weight.requires_grad_(False)
        else:
            print('Error: The shape of the given weights is not correct')
    
    # Note that a second way to define the conv. layer here would be to pass group = 3 when calling torch.nn.Conv2d
    
    def forward(self, x):
        return self.conv(x)



# function to massage img_tensor for using as input to plt.imshow()
def image_converter(im):
    # undo the above normalization if and when the need arises 
    denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225] )
    # move the image to cpu
    im_copy = im.cpu()    
    # for plt.imshow() the channel-dimension is the last
    # therefore use transpose to permute axes
    im_copy = denormalize(im_copy.clone().detach()).numpy()
    im_copy = im_copy.transpose(1,2,0)
    
    # clip negative values as plt.imshow() only accepts 
    # floating values in range [0,1] and integers in range [0,255]
    im_copy = im_copy.clip(0, 1) 
    
    return im_copy




# function to compute gradient loss of an image 
def grad_loss(img, gradLayer,beta = 1, device = 'cpu'):
    
    # move the gradLayer to cuda
    gradLayer.to(device)
    gradSq = gradLayer(img.unsqueeze(0))**2
    
    grad_loss = torch.pow(gradSq.mean(), beta/2)
    
    return grad_loss

class FeatureVisualization():
    def __init__(self,model) -> None:
        self.model = model
    def visualizeFeature(self,filter_id, layer_name,init_dim = 56,upscaling_steps=12,optim_steps=20,act_wt = 0.5,upscaling_factor =  1.2):
        for param in self.model.parameters():
            param.requires_grad_(False)
        print(list(map(lambda x: x[0], self.model.named_children())))

        # register a forward hook for layer inception4a
        # model.inception4a.register_forward_hook(create_hook('4a'))
        # model.maxpool2.register_forward_hook(create_hook(layer_name))
        self.model.features[6].register_forward_hook(create_hook(layer_name))

        # normalize the input image to have appropriate mean and standard deviation as specified by pytorch
        from torchvision import transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])



        # generate a numpy array with random values
        img = np.single(np.random.uniform(0,1, (3, init_dim, init_dim)))
        # convert to a torch tensor, normalize, set the requires_grad_ flag
        im_tensor = normalize(torch.from_numpy(img)).requires_grad_(True)
        # plt.imshow(image_converter(im_tensor))
        # # plt.show()
        # plt.title('Initial image')
        # plt.savefig(f"./results/{net_name}_{layer_name}_{unit_idx}_{Width}_scale_0.png")

        # Scharr Filters
        filter_x = np.array([[-3, 0, 3], 
                            [-10, 0, 10],
                            [-3, 0, 3]])
        filter_y = filter_x.T
        grad_filters = np.array([filter_x, filter_y])

        gradLayer = RGBgradients(grad_filters)

        # move everything to the GPU -> can be skept
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Calculations being executed on {}'.format(device))
        self.model.to(device)
        img_tensor = im_tensor.to(device)




        # unit_idx = 162 # the neuron to visualize
        # act_wt = 0.5 # factor by which to weigh the activation relative to the regulizer terms
        # upscaling_steps = 45 # no. of times to upscale
        # upscaling_factor = 1# 1.05
        # optim_steps = 20# no. of times to optimize an input image before upscaling

        self.model.eval()
        for mag_epoch in range(upscaling_steps):
            clone_img_tensor = img_tensor.clone().detach()
            clone_img_tensor = clone_img_tensor.requires_grad_(True)
            # optimizer = optim.Adam([img_tensor], lr = 0.4)
            optimizer = optim.Adam([clone_img_tensor], lr = 0.4)
            
            for opt_epoch in range(optim_steps):
                optimizer.zero_grad()
                # model(img_tensor.unsqueeze(0))
                self.model(clone_img_tensor.unsqueeze(0))
                # layer_out = activation['4a']
                layer_out = activation[layer_name]
                rms = torch.pow((layer_out[0, filter_id]**2).mean(), 0.5)
                # terminate if rms is nan
                if torch.isnan(rms):
                    print('Error: rms was Nan; Terminating ...')
                    sys.exit()
                
                # pixel intensity
                pxl_inty = torch.pow((clone_img_tensor**2).mean(), 0.5)
                # terminate if pxl_inty is nan
                if torch.isnan(pxl_inty):
                    print('Error: Pixel Intensity was Nan; Terminating ...')
                    sys.exit()
                    
                # image gradients
                im_grd = grad_loss(clone_img_tensor,gradLayer, beta = 1, device = device)
                # terminate is im_grd is nan
                if torch.isnan(im_grd):
                    print('Error: image gradients were Nan; Terminating ...')
                    sys.exit()
                
                loss = -act_wt*rms + pxl_inty + im_grd        
                # print activation at the beginning of each mag_epoch
                if opt_epoch == 0:
                    print('begin mag_epoch {}, activation: {}'.format(mag_epoch, rms))
                loss.backward()
                optimizer.step()
                
            # view the result of optimising the image
            print('end mag_epoch: {}, activation: {}'.format(mag_epoch, rms))
            img = image_converter(clone_img_tensor)    
            plt.imshow(img)
            plt.title('image at the end of mag_epoch: {}'.format(mag_epoch+1))
            # plt.show()
            # plt.savefig(f"./results/{net_name}_{layer_name}_{unit_idx}_{Width}_scale_{mag_epoch+1}.png")#for giff generation
            net_name = self.model.__class__.__name__ 
            plt.savefig(f"./results/{net_name}_{layer_name}_{filter_id}_{init_dim}_scale.png")
            
            img = cv2.resize(img, dsize = (0,0), 
                            fx = upscaling_factor, fy = upscaling_factor).transpose(2,0,1) # scale up and move the batch axis to be the first
            clone_img_tensor = normalize(torch.from_numpy(img)).to(device).requires_grad_(True)
            img_tensor = clone_img_tensor
        return 0
        
        

##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%
##%%##%%##%%##%%#           main             #%%##%%##%%##%%##%%
##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%##%%
def main():

    model = models.vgg16(pretrained = True)
    fv = FeatureVisualization(model)
    fv.visualizeFeature(112,'4a')

if __name__=="__main__":
    main()
