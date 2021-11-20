import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Loading The DataSet
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)

train_data_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)

# Defining our AutoEncoder Architecture 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder , self).__init__()        
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16) ,
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded_data = self.encoder(x)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

# Training The Model 

model = Autoencoder()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)

number_of_epochs = 10
results = []
for epoch in range(number_of_epochs):
    for (img, _) in train_data_loader:
        img = img.reshape(-1 , 784) 
        reconstructed_image = model(img)
        loss = loss_function(reconstructed_image, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch : {epoch+1} , Loss : {loss.item()} ')
    results.append((epoch, img, reconstructed_image))

# Visualize The Result 

def plot_result(autoEncoder_result , nbr_epoch):
    original_images = autoEncoder_result[ nbr_epoch -1 ][1].detach().numpy()
    reconstructed_images = autoEncoder_result[ nbr_epoch -1 ][2].detach().numpy()

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