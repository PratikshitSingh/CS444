from __future__ import barry_as_FLUFL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img
from torchvision import torch

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Train loop for GAN.
    
    The loop will consist of two steps: a discriminator step and a generator step.
    
    (1) In the discriminator step, you should zero gradients in the discriminator 
    and sample noise to generate a fake data batch using the generator. Calculate 
    the discriminator output for real and fake data, and use the output to compute
    discriminator loss. Call backward() on the loss output and take an optimizer
    step for the discriminator.
    
    (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.
    
    You will need to reshape the fake image tensor outputted by the generator to 
    be dimensions (batch_size x input_channels x img_size x img_size).
    
    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations.
    
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))
        for x, _ in train_loader:
            #print(x.shape) # this is batch * channels = 1 in the case of black and white * H * W
            _, input_channels, img_size, _ = x.shape
            x = x.to(device)
            #print(_, input_channels, img_size, _)
            real_images = preprocess_img(x).to(device)  # normalize
            
            # Store discriminator loss output, generator loss output, and fake image output
            # in these variables for logging and visualization below
            d_error = None
            g_error = None
            fake_images_from_G = None
            
            ####################################
            #          YOUR CODE HERE          #
            ####################################
            # we make a noise sample 
            batch_size_for_noise = batch_size
            sample_noise_ = sample_noise(batch_size_for_noise, noise_size).to(device)
            #sample_noise_ = torch.reshape(sample_noise_, (batch_size_for_noise, noise_size, 1, 1))
            #remove the previous line for MNIST
            #print(sample_noise_)
            # we give it to the generator , which gives us the image generated, these are fake
            fake_images_from_G = G(sample_noise_)
            #print(fake_images_from_G.shape)
            fake_images_from_G = torch.reshape(fake_images_from_G, (batch_size_for_noise , input_channels, img_size, img_size))
            # we give that to the discriminator , it should return fake as output
            output_of_discriminator_from_fake_images = D(fake_images_from_G)
            # we pass the real images into the discriminator - the x here, discrimnator should return 1 for these. 
            
            output_of_discriminator_from_real_images = D(x)
            # we use discriminator loss and store it in the loss 
            d_error = discriminator_loss(output_of_discriminator_from_real_images , output_of_discriminator_from_fake_images)

            # zero grad and back ward
            D_solver.zero_grad()
            d_error.backward()
            D_solver.step()
            # and we are done with step 1 of the train 

            # now that a random sample and pass through the genrator, these are fake images
            batch_size_for_noise = batch_size
            sample_noise_ = sample_noise(batch_size_for_noise, noise_size).to(device)
            #sample_noise_ = torch.reshape(sample_noise_, (batch_size_for_noise, noise_size, 1, 1))
            #remove the previous line for MNIST
            fake_images_from_G = G(sample_noise_)
            fake_images_from_G = torch.reshape(fake_images_from_G, (batch_size_for_noise , input_channels, img_size, img_size))
            
            # pass them through discriminator 1
            output_of_discriminator_from_fake_images = D(fake_images_from_G)
            # then see the result of the discriminator 
            # pass that throught the generator loss funciton 
            g_error = generator_loss(output_of_discriminator_from_fake_images)

            # zero grad and backward
            G_solver.zero_grad()
            g_error.backward()
            G_solver.step()
            # and youre done . 
            
            ##########       END      ##########
            
            # Logging and output visualization
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_error.item(),g_error.item()))
                disp_fake_images = deprocess_img(fake_images_from_G.data)  # denormalize
                imgs_numpy = (disp_fake_images).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1