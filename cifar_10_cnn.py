import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataLoader for CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create directories to save generated images
os.makedirs('generated_images/cifar_cnn', exist_ok=True)

# Generator Network (using CNN)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False), # Latent space to larger feature map
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # Upsample to 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # Upsample to 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), # Upsample to 32x32 (RGB image)
            nn.Tanh()  # To ensure the values are in the range [-1, 1]
        )

    def forward(self, z):
        return self.main(z)

# Discriminator Network (using CNN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)  # 8x8 -> 4x4
        self.fc = nn.Linear(256 * 4 * 4, 1)  # Final fully connected layer to get scalar output

    def forward(self, x):
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv1(x))
        x = nn.BatchNorm2d(128)(nn.LeakyReLU(0.2, inplace=True)(self.conv2(x)))
        x = nn.BatchNorm2d(256)(nn.LeakyReLU(0.2, inplace=True)(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)  # Flatten the tensor
        x = self.fc(x)
        return torch.sigmoid(x)  # Return a scalar value (real/fake probability)

# Initialize networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Helper function to save generated images and comparison with real images
def save_generated_images(epoch, real_images, fake_images):
    # Plot real images
    real_grid = torchvision.utils.make_grid(real_images, nrow=8, normalize=True)
    # Plot fake images
    fake_grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
    
    # Create a side-by-side comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Show real images
    ax[0].imshow(np.transpose(real_grid, (1, 2, 0)))
    ax[0].axis('off')
    ax[0].set_title("Real Images")

    # Show fake images
    ax[1].imshow(np.transpose(fake_grid, (1, 2, 0)))
    ax[1].axis('off')
    ax[1].set_title("Generated Images")

    # Save the comparison image
    plt.savefig(f'generated_images/cifar_cnn/epoch_{epoch}.png')
    plt.close()

# Training loop
num_epochs = 20
fixed_noise = torch.randn(64, 100, 1, 1, device=device)  # To generate consistent images for each epoch

total_start_time = time.time()  # Start time for the entire training process

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Start time for the current epoch
    
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Create labels for real and fake images
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        optimizer_d.zero_grad()
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(batch_size, 100, 1, 1, device=device)  # Latent vector (1x1 feature map)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())  # Detach fake images for discriminator
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()

        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # We want to fool the discriminator
        g_loss.backward()

        optimizer_g.step()

        # Print progress
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'D Loss: {d_loss_real.item() + d_loss_fake.item():.4f}, G Loss: {g_loss.item():.4f}')

    # Save generated images after every epoch with comparison
    save_generated_images(epoch+1, real_images.cpu(), fake_images.detach().cpu())

    # Calculate time taken for this epoch
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    remaining_epochs = num_epochs - (epoch + 1)
    avg_epoch_time = (epoch_end_time - total_start_time) / (epoch + 1)
    remaining_time = avg_epoch_time * remaining_epochs

    # Print time estimates
    print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
    print(f"Estimated time remaining: {remaining_time / 60:.2f} minutes.")

print('Training Finished.')
