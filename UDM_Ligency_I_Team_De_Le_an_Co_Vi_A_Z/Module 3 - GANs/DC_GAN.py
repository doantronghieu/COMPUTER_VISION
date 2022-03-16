# Libraries
from   __future__ import  print_function
import enum
from cv2 import normalize
import torch
import torch.nn               as nn
import torch.nn.parallel
import torch.optim            as optim
import torch.utils.data
import torchvision.datasets   as dset
import torchvision.transforms as transforms
import torchvision.utils      as vutils
from   torch.autograd         import  Variable

# Hyperparameters
batch_size = 64
image_size = 64 # Size of the generated image (64x64)

# Creating the transformation
# Make the input images compatible with the neural network of the generator
transform = transforms.Compose([transforms.Scale(image_size), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
# Use dataLoader to get the images of the training set batch by batch.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                        shuffle = True, num_workers = 2)

# Takes as input a neural network m and initialize all its weights
# Initialize all the weight of the neural network (generator and discriminator)
def weights_init(m):
    # look for some names in the definition of the class
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator
class G(nn.Module):
    def __init__(self):
        # Activate the inheritance
        super(G, self).__init__()
        
        # Make a meta module (all modules in a sequence of layers)
        self.main = nn.Sequential(
            # inverse of a convolution
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            # normalize all the features along the dimension of the batch
            # between minus one and plus one, centered around zero
            # want the same standards as the images of the data set
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # Add non-linearity
            nn.Tanh()
        )
        
    # Propagate the signal inside the whole neural network of the generator
    # Input: Random vector noise
    # Output: 3-channel fake image
    def forward(self, input):
        output = self.main(input)
        return output

# Creating the generator
net_G = G()
net_G.apply(weights_init)

# Defining the discriminator
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # "Discriminator"
            nn.Sigmoid()
        )
    
    def forward(self, input):
        """
        - The discriminator takes as input an image created by the generator,
        it will decide if it wants to accept or reject the image.
        - returns the output that is discriminating value between zero and one.
        - if this output is close to zero, it will reject.
        - if this output is close to one, it will accept.
        """
        output = self.main(input)
        return output.view(-1) # flatten the result of the convolutions into 1 vector

# Creating the Discriminator
net_D = D()
net_D.apply(weights_init)

""" Training the DC GANs
- train the discriminator to understand what's real and what's fake
- giving it a real image and we will set the target to one
- giving it a fake image and we will set the target to zero (created by the generator) 
- BCE: Binary Cross Entropy """
criterion   = nn.BCELoss()
optimizer_D = optim.Adam(params=net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G = optim.Adam(params=net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))

"""
- updating the weight of the neural network of the generator
- take the fake image, feed this fake image into the discriminator to get the output,
(0->1), set a new target to one, compute the loss between the output of the discriminator
and this target (1), backprop this error inside the G, SGD update the weights of the G """
for epoch in range(25):
    # go through all the images of the data set (each minibatch)
    for i, data in enumerate(dataloader, start=0):
        # 1st step: Updating the weights of the NN of the D
        # initialize the gradient of the discriminator with respect to the weight to zero
        net_D.zero_grad()
        
        # train the discriminator to see and understand what's real and what's fake
        # Training the D with a real image of the dataset
        real, _ = data # Get the real image
        input   = Variable(real) # Convert to Torch Variable
        # specify to the discriminator that the ground truth is actually one (real img)
        # input.size()[0]: the size of the minibatch
        target       = Variable(torch.ones(size=(input.size()[0])))
        output       = net_D(input)
        err_D_real   = criterion(output, target)
        
        # Training the D with a fake image generated by the generator
        # Feed the neural network of the generator with random vector 
        # createa mini batch of random vectors of size 100
        # 1, 1: fake dimensions that will correspond to a 100 feature map of size 1x1
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        # New input of Ground truth: Fake imgs
        fake       = net_G(noise)
        target     = Variable(torch.zeros(size=(input.size()[0])))
        output     = net_D(fake.detach()) # detach(): don't care the gradient
        err_D_fake = criterion(output, target)
        
        # Backpropagating the total error into the neural network of the discriminator
        err_D = err_D_real + err_D_fake
        err_D.backward()
        """ apply stochastic gradient descent to update the weight of the discriminator
        according to how much they're responsible for the total loss error """
        optimizer_D.step()
        
        # 2nd step: Updating the weights of the NN of the G
        """ the error between the prediction of the discriminator, whether
        or not the image generated by the generator should be accepted
        - Target = 1: Want the G to have some weights to produce img that look like
        real img -> Push the predict close to 1
        - Push the prediction to 1, want discriminator to accept that the fake images 
        are real images
        - Keep the gradient of fake to update the weights """
        # initializing the gradient of the generator with respect to the weights to zero
        net_G.zero_grad()
        target = Variable(torch.ones(size=(input.size()[0])))
        output = net_D(fake)
        err_G  = criterion(output, target)
        err_G.backward()
        optimizer_G.step() # Update the weights
        
        # 3rd step: Printing the Losses & saving the real imgs & the generated imgs of
        #  the minibatch every 100 steps
        print(f'[{epoch}/25][{i}/{len(dataloader)}] Loss_D: {err_D.data[0]:.4f}, Loss_G: {err_G.data[0]:.4f}')
        
        if (i % 100 == 0):
            vutils.save_image(real, r'C:\Users\Doan Trong Hieu\Downloads\IMPORTANT\SPECIALIZATION\Artificial_Intelligence\COMPUTER VISION\CODING_COMPUTER_VISION\UDM_Ligency_I_Team_De_Le_an_Co_Vi_A_Z\Module 3 - GANs\results\real_samples.png',
                            normalize=True)
            fake = net_G(noise)
            vutils.save_image(fake.data, r'C:\Users\Doan Trong Hieu\Downloads\IMPORTANT\SPECIALIZATION\Artificial_Intelligence\COMPUTER VISION\CODING_COMPUTER_VISION\UDM_Ligency_I_Team_De_Le_an_Co_Vi_A_Z\Module 3 - GANs\results\fake_samples_epoch_%03d.png' % epoch,
                            normalize=True)            