from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = torch.rand((batch_size, noise_dim), dtype=dtype, device=device)
    # interpolate between -1 and 1 -> noise closer to 0 will make results close to 1,
    # noise closer to 1 will make results close to -1, noise close to 0.5 will
    # make results close to 0. 
    noise = -1 * noise + 1 * (1-noise)
    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, 256),
                          nn.LeakyReLU(negative_slope=0.01),
                          nn.Linear(256, 256),
                          nn.LeakyReLU(negative_slope=0.01),
                          nn.Linear(256, 1))
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = nn.Sequential(nn.Linear(noise_dim, 1024),
                          nn.ReLU(),
                          nn.Linear(1024, 1024),
                          nn.ReLU(),
                          nn.Linear(1024, 784),
                          nn.Tanh())
    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    #  torch.ones(logits_real.shape) because real labels are 1
    ld_data = torch.nn.functional.binary_cross_entropy_with_logits(logits_real, torch.ones(logits_real.shape)) 
    #  torch.zeros(logits_fake.shape) because fake labels are 0
    ld_gen = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.zeros(logits_fake.shape))
    # Sum "real" and "fake" losses.
    # That is, BCE has already taken into account the "negated equation" form as it applies minus sign in the formula (see notebook),
    # the "log" (in the Expectation) as we pass the scores and it applies sigmoid (to get probabilities) and log and the "mean" (see "reduction method").
    loss = ld_gen + ld_data
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    # For the generator (G), the true target (y = 1) corresponds to "fake" images.
    # Thus, for the scores of fake images, the target is always 1 (a vector).
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.ones(logits_fake.shape))
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    real_loss = 0.5 * ((scores_real-1)**2).mean()
    fake_loss = 0.5 * (scores_fake**2).mean()
    loss = real_loss + fake_loss
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    fake_loss = 0.5 * (scores_fake-1)**2
    loss = fake_loss.mean()
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = nn.Sequential(
        # Unflatten the model's input. Output shape is (batch_size, 1, 28, 28)
        nn.Unflatten(1, (1, 28, 28)),
        # Apply Conv2D layer and LeakyReLU. Output shape is (batch_size, 32, 26, 26)
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
        nn.LeakyReLU(0.01),
        # Apply Max Pooling. Output shape is (batch_size, 32, 13, 13)
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Apply Conv2D layer and LeakyReLU. Output shape is (batch_size, 64, 11, 11)
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
        nn.LeakyReLU(0.01),
        # Apply Max Pooling. Output shape is (batch_size, 64, 4, 4)
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Flatten the data. Output shape is (batch_size, 64*4*4)
        nn.Flatten(),
        # Apply FC layer and LeakyReLU. Output shape is (batch_size, 64*4*4)
        nn.Linear(4*4*64, 4*4*64),
        nn.LeakyReLU(0.01),
        # Apply FC layer (Output layer). Output shape is (batch_size, 1)
        nn.Linear(4*4*64, 1)
    )
    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = nn.Sequential(
            # Apply FC layer, ReLU and Batch norm. Output shape is (1, 1024)
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            # Apply FC layer, ReLU and Batch norm. Output shape is (1, 7*7*128)
            nn.Linear(1024, 7*7*128),
            nn.ReLU(),
            nn.BatchNorm1d(7*7*128),
            # Reshape the data into Image Tensor. Output shape is (128, 7, 7)
            nn.Unflatten(1, (128, 7, 7)),
            # Apply Conv2D Transpose layer, ReLU and Batch norm.
            # Note that in PyTorch, the padding-type in this layer type must be 'zero'
            # (default value), 'same' padding is not permitted. Output shape is (64, 14, 14)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Apply Conv2D Transpose and TanH. Output shape is (1, 28, 28)
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4,
                          stride=2, padding=1),
            nn.Tanh(),
            # Flatten the data. Output shape is (784,)
            nn.Flatten()
          )

    return model
