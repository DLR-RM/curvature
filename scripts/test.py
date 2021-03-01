"""This is the code featured in the 60-seconds blitz to Laplace approximation found in the readme."""

# Standard imports
import torch
import torchvision
import tqdm

# From the repository
from curvature.curvatures import KFAC
from curvature.lenet5 import lenet5


# Change this to 'cuda' if you have a working GPU.
device = 'cpu'

# We will use the provided LeNet-5 variant pre-trained on MNIST.
model = lenet5(pretrained='mnist', device=device).to(device)

# Standard PyTorch dataset location
torch_data = "~/.torch/datasets"
mnist = torchvision.datasets.MNIST(root=torch_data,
                                   train=True,
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True)
train_data = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)

# Decide which loss criterion and src approximation to use.
criterion = torch.nn.CrossEntropyLoss().to(device)
kfac = KFAC(model)

# Standard PyTorch training loop:
for images, labels in tqdm.tqdm(train_data):
    logits = model(images.to(device))

    # We compute the 'true' Fisher information matrix (FIM),
    # by taking the expectation over the model distribution.
    # To obtain the empirical FIM, just use the labels from
    # the data distribution directly.
    dist = torch.distributions.Categorical(logits=logits)
    sampled_labels = dist.sample()

    loss = criterion(logits, sampled_labels)
    model.zero_grad()
    loss.backward()

    # We call 'estimator.update' here instead of 'optimizer.step'.
    kfac.update(batch_size=images.size(0))

# Invert the src information to perform Bayesian inference.
# 'Add' and 'multiply' are the two regularization hyperparameters of Laplace approximation.
# Please see the tutorial notebook for for in-depth examples and explanations.
kfac.invert(add=0.5, multiply=1)
kfac.sample_and_replace()
