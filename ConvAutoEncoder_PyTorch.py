import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from torch.autograd import Variable
import matplotlib.pyplot as plt


# Loading The Dataset and Creating The Data Loader to iterate Over The Data
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)

train_data_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)

# defining our Convolution Autoencoder Architecture

class Convolution_Autoencoder(nn.Module):
    def __init__(self):
        super(Convolution_Autoencoder , self).__init__()        
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3 , padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)),
            nn.Conv2d(16, 32, 3, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2)),
            nn.Conv2d(32, 64, 6) 
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded_data = self.encoder(x)
        decoded_data = self.decoder(encoded_data)
        return decoded_data
    
# visualizing The Architecture
model = Convolution_Autoencoder().cuda()
summary(model , (1 , 28 , 28))
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)

# Training The Model
number_of_epochs = 10
results = []
for epoch in range(number_of_epochs):
    for (img, _) in train_data_loader:
        #img = img.reshape(-1 , 784) 
        img = Variable(img).cuda()
        reconstructed_image = model(img)
        loss = loss_function(reconstructed_image, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    results.append((epoch, img, reconstructed_image))

# Visualizing The Result
def plot_result(autoEncoder_result , nbr_epoch):
    original_images = autoEncoder_result[ nbr_epoch -1 ][1].cpu().data.numpy()
    reconstructed_images = autoEncoder_result[ nbr_epoch -1 ][2].cpu().data.numpy()

    # Plot The Original Images 
    plt.figure(figsize=(9, 3))
    plt.gray()
    for index, image in zip(range(9) , original_images):
            plt.subplot(2, 9, index+1)
            image = image.reshape(-1, 28,28)
            plt.imshow(image[0])
    plt.suptitle(f"Original Images " , fontsize = 20) 

    # Plot The Reconstructed images
    plt.figure(figsize=(9, 3))
    plt.gray()
    for index, image in zip(range(9), reconstructed_images):
        plt.subplot(2, 9, index+1)
        image = image.reshape(-1, 28,28) 
        plt.imshow(image[0])        
    plt.suptitle(" Reconstructed Images" , fontsize = 20)   

plot_result(results , number_of_epochs)