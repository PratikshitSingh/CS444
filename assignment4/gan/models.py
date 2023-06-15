import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1      = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv2      = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn1        = nn.BatchNorm2d(256)
        self.conv3      = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn2        = nn.BatchNorm2d(512)
        self.conv4      = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.bn3        = nn.BatchNorm2d(1024)
        self.conv5      = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.LeakyReLU  = nn.LeakyReLU(0.2)
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.LeakyReLU(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.LeakyReLU(x)
        x = self.conv5(x)
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1      = nn.ConvTranspose2d(in_channels=noise_dim, out_channels=1024, kernel_size=4, stride=1)
        self.bn1        = nn.BatchNorm2d(1024)
        self.conv2      = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2,  padding =1)
        self.bn2        = nn.BatchNorm2d(512)
        self.conv3      = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding =1)
        self.bn3        = nn.BatchNorm2d(256)
        self.conv4      = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2 ,padding =1)
        self.bn4        = nn.BatchNorm2d(128)
        self.conv5      = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding =1)
        self.tanh       = nn.Tanh()
        self.LeakyReLU  = nn.LeakyReLU(0.2)
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.LeakyReLU(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.LeakyReLU(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.LeakyReLU(x)
        x = self.conv5(x)
        x = self.tanh(x)
        
        ##########       END      ##########
        
        return x
    

