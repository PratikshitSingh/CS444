import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    real_labels = torch.ones(logits_real.size()).cuda()
    fake_labels = torch.zeros(logits_fake.size()).cuda()
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss_real = bce_loss(logits_real, real_labels)
    loss_fake = bce_loss(logits_fake, fake_labels)
    loss      = loss_real + loss_fake
    
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    real_labels = torch.ones(logits_fake.size()).cuda()
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss  = bce_loss(logits_fake, real_labels)
    
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    real_labels = torch.ones(scores_real.size()).cuda()
    fake_labels = torch.zeros(scores_fake.size()).cuda()
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss_real = torch.mean((scores_real - real_labels) ** 2)
    loss_fake = torch.mean((scores_fake - fake_labels) ** 2)
    loss      = loss_real + loss_fake
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    real_labels = torch.ones(scores_fake.size()).cuda()
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss  = torch.mean((scores_fake - real_labels) ** 2)
    
    ##########       END      ##########
    
    return loss
