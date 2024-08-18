from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

import torchvision.datasets as TVdatasets
from torchvision import transforms as TVtransforms


EXP_NAME = "ns_cond_gen_cond_desc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
DISCRIMINATOR_STEPS = 1
LR = 2e-4
N_EPOCHS = 100


def plot_images(batch, imname, title=None):
    images, labels = batch
    n_plots = len(labels)
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots, 1), dpi=100)
    for i in range(n_plots):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
        axes[i].set_title(f"label: {label}")
    if title is not None:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(pad=0.5)
    plt.savefig(imname)
    plt.close()


class MNIST(torch.utils.data.Dataset):

    def __init__(
        self,
        train: bool = True,
        spat_dim=(16, 16),
        download: bool = False,
        pix_range=(-1.0, 1.0),
    ) -> None:
        _m, _std = pix_range[0] / float(pix_range[0] - pix_range[1]), 1.0 / float(
            pix_range[1] - pix_range[0]
        )
        TRANSFORM = TVtransforms.Compose(
            [
                TVtransforms.Resize(spat_dim),
                TVtransforms.ToTensor(),
                TVtransforms.Normalize([_m], [_std]),
            ]
        )
        self.mnist = TVdatasets.MNIST(
            root="./data", train=train, download=download, transform=TRANSFORM
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.mnist[idx]


class ConditionalConvGenerator(nn.Module):

    def __init__(
        self,
        input_size: int = 128,
    ) -> None:
        super().__init__()
        self.n_channels = 64
        self.input_size = input_size
        self.linear_block = nn.Sequential(
            nn.Linear(input_size + 10, 4 * 4 * 4 * self.n_channels),
            nn.ReLU(True),
        )
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(4 * self.n_channels, 2 * self.n_channels, 2, stride=2),
            nn.BatchNorm2d(2 * self.n_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * self.n_channels, self.n_channels, 2, stride=2),
            nn.BatchNorm2d(self.n_channels),
            nn.ReLU(True),
            nn.Conv2d(self.n_channels, 1, 3, padding=1),
        )
        self.noise = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, noise: torch.Tensor, cls_embedding: torch.Tensor) -> torch.Tensor:
        assert noise.size(1) == self.input_size
        cond_noise = torch.cat((noise, cls_embedding), dim=1)
        output = self.linear_block(cond_noise)
        output = output.view(-1, 4 * self.n_channels, 4, 4)
        output = self.conv_block(output)
        output = torch.tanh(output)
        return output.view(-1, 1, 16, 16)

    def sample(self, n_samples: int, cls_embedding: torch.Tensor) -> torch.Tensor:
        z = self.noise.sample([n_samples, self.input_size]).to(
            next(iter(self.parameters()))
        )
        return self.forward(z, cls_embedding)


class ConvDiscriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.n_channels = 64

        self.net = nn.Sequential(
            nn.Conv2d(1, self.n_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.n_channels, 2 * self.n_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.n_channels, 4 * self.n_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(4 * 2 * 2 * self.n_channels, 1)

    def forward(self, x: torch.Tensor, _) -> torch.Tensor:
        output = self.net(x)
        output = output.view(-1, 4 * 2 * 2 * self.n_channels)
        output = self.linear(output)
        return output


class VanillaConvDiscriminator(ConvDiscriminator):

    def forward(self, x: torch.Tensor, _) -> torch.Tensor:
        output = super().forward(x)
        return torch.sigmoid(output)


class ConditionalConvDiscriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.n_channels = 64

        self.net = nn.Sequential(
            nn.Conv2d(1, self.n_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.n_channels, 2 * self.n_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.n_channels, 4 * self.n_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(4 * 2 * 2 * self.n_channels + 10, self.n_channels),
            nn.ReLU(),
            nn.Linear(self.n_channels, 1),
        )

    def forward(self, x: torch.Tensor, cls_embedding: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        output = output.view(-1, 4 * 2 * 2 * self.n_channels)
        output = torch.cat((output, cls_embedding), dim=1)
        output = self.mlp(output)
        return output


class ConditionalVanillaConvDiscriminator(ConditionalConvDiscriminator):

    def forward(self, x: torch.Tensor, cls_embedding: torch.Tensor) -> torch.Tensor:
        output = super().forward(x, cls_embedding)
        return torch.sigmoid(output)


def vanilla_gen_step(
    X: torch.Tensor,
    classes: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
) -> torch.Tensor:

    G.train()
    D.eval()
    batch_size = X.size(0)
    X_gen = G.sample(batch_size, classes)
    scores_gen = D(X_gen, classes)
    loss = -F.binary_cross_entropy(scores_gen, torch.zeros_like(scores_gen))
    G_optim.zero_grad()
    loss.backward()
    G_optim.step()
    return loss.item()


def vanilla_discr_step(
    X: torch.Tensor,
    classes: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    D_optim: torch.optim.Optimizer,
) -> torch.Tensor:

    G.eval()
    D.train()
    batch_size = X.size(0)
    with torch.no_grad():
        X_gen = G.sample(batch_size, classes)
    scores_gen = D(X_gen, classes)
    scores_real = D(X, classes)
    loss_gen = F.binary_cross_entropy(scores_gen, torch.zeros_like(scores_gen))
    loss_real = F.binary_cross_entropy(scores_real, torch.ones_like(scores_real))
    loss = loss_gen + loss_real

    D_optim.zero_grad()
    loss.backward()
    D_optim.step()

    return loss.item()


def train_vanilla(
    train_loader: DataLoader,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
    D_optim: torch.optim.Optimizer,
    discriminator_steps: int,
    n_epochs: int,
    # diagnostic : GANDiagnosticCompanion,
    visualize_steps: int = 10,
) -> None:

    G.train()
    D.train()
    step_i = 0
    for epoch_i in tqdm(range(n_epochs)):
        for batch_i, X in enumerate(train_loader):
            X, classes = X

            X = X.to(DEVICE)
            classes = F.one_hot(classes, num_classes=10).type(torch.float).to(DEVICE)

            d_loss = vanilla_discr_step(X, classes, G, D, D_optim)
            wandb.log(
                {
                    "discriminator_loss": d_loss,
                }
            )

            # GENERATOR UPDATE
            if step_i % discriminator_steps == 0:
                g_loss = vanilla_gen_step(X, classes, G, D, G_optim)
                wandb.log(
                    {
                        "generator_loss": g_loss,
                    }
                )

            step_i += 1

        if visualize_steps and epoch_i % visualize_steps == 0:
            targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            G.eval()
            result = (
                G.sample(
                    10, F.one_hot(targets, num_classes=10).type(torch.float).to(DEVICE)
                )
                .detach()
                .cpu()
                .numpy()
            )
            plot_images((result, targets), f"media/{EXP_NAME}.png")

    torch.save(G.state_dict(), f"weights/G_{EXP_NAME}.pth")
    torch.save(D.state_dict(), f"weights/D_{EXP_NAME}.pth")


def ns_gen_step(
    X: torch.Tensor,
    classes: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
) -> torch.Tensor:
    G.train()
    D.eval()
    batch_size = X.size(0)
    X_gen = G.sample(batch_size, classes)
    scores_gen = D(X_gen, classes)
    loss = F.softplus(-scores_gen).mean()
    G_optim.zero_grad()
    loss.backward()
    G_optim.step()
    return loss.item()


def ns_discr_step(
    X: torch.Tensor,
    classes: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    D_optim: torch.optim.Optimizer,
    r1_regularizer: float = 1.0,
) -> torch.Tensor:
    G.eval()
    D.train()
    D_optim.zero_grad()
    batch_size = X.size(0)
    with torch.no_grad():
        X_gen = G.sample(batch_size, classes)
    X.requires_grad_()
    scores_gen = D(X_gen, classes)
    scores_real = D(X, classes)
    loss_gen = F.softplus(scores_gen).mean()
    loss_real = F.softplus(-scores_real).mean()
    scores_real.sum().backward(retain_graph=True, create_graph=True)
    gradients = X.grad
    grad_penalty = (gradients.view(gradients.size(0), -1).norm(2, dim=1) ** 2).mean()
    D_optim.zero_grad()
    loss = loss_gen + loss_real + r1_regularizer * grad_penalty

    loss.backward()
    D_optim.step()
    gradients.detach_()  # to avoid memory leak!
    return loss.item()


def train_ns(
    train_loader: DataLoader,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
    D_optim: torch.optim.Optimizer,
    discriminator_steps: int,
    n_epochs: int,
    r1_regularizer: float = 1.0,
    visualize_steps: int = 10,
) -> None:

    G.train()
    D.train()
    step_i = 0
    for epoch_i in tqdm(range(n_epochs)):
        for batch_i, X in enumerate(train_loader):
            X, classes = X

            X = X.to(DEVICE)
            classes = F.one_hot(classes, num_classes=10).type(torch.float).to(DEVICE)

            d_loss = ns_discr_step(X, classes, G, D, D_optim, r1_regularizer)
            wandb.log(
                {
                    "discriminator_loss": d_loss,
                }
            )

            if step_i % discriminator_steps == 0:
                g_loss = ns_gen_step(X, classes, G, D, G_optim)
                wandb.log(
                    {
                        "generator_loss": g_loss,
                    }
                )

            step_i += 1

        if visualize_steps and epoch_i % visualize_steps == 0:
            targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            G.eval()
            result = (
                G.sample(
                    10, F.one_hot(targets, num_classes=10).type(torch.float).to(DEVICE)
                )
                .detach()
                .cpu()
                .numpy()
            )
            plot_images((result, targets), f"media/{EXP_NAME}.png")

    torch.save(G.state_dict(), f"weights/G_{EXP_NAME}.pth")
    torch.save(D.state_dict(), f"weights/D_{EXP_NAME}.pth")


def main():
    mnist_train = MNIST(train=True, download=True)
    mnist_test = MNIST(train=False)

    plot_images(next(iter(DataLoader(mnist_train, batch_size=10))), "media/mnist_example.png")

    train_mnist_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
    G = ConditionalConvGenerator().to(DEVICE)
    # D = ConvDiscriminator().to(DEVICE)
    # D = VanillaConvDiscriminator().to(DEVICE)
    D = ConditionalConvDiscriminator().to(DEVICE)
    # D = ConditionalVanillaConvDiscriminator().to(DEVICE)
    G_optim = torch.optim.RMSprop(G.parameters(), lr=LR)
    D_optim = torch.optim.RMSprop(D.parameters(), lr=LR)

    wandb.init(project="SMILES-GANs", name=EXP_NAME)

    # train_vanilla(
    #     train_mnist_loader,
    #     G,
    #     D,
    #     G_optim,
    #     D_optim,
    #     discriminator_steps=DISCRIMINATOR_STEPS,
    #     n_epochs=N_EPOCHS,
    #     visualize_steps=1
    # )

    train_ns(
        train_mnist_loader,
        G,
        D,
        G_optim,
        D_optim,
        discriminator_steps=DISCRIMINATOR_STEPS,
        n_epochs=N_EPOCHS,
        r1_regularizer=0.1,
        visualize_steps=1,
    )


if __name__ == "__main__":
    main()
