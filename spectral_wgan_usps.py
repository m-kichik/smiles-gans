from typing import Optional

import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as TVdatasets
from torchvision import transforms as TVtransforms

from conditional_mnist_conv import plot_images


EXP_NAME = "wgan_2_power_it_1000_lr-3_c1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
CRITIC_STEPS = 1
LR = 2e-3
N_EPOCHS = 1000
POWER_IT = 2


def power_iteration_method(
    W: torch.Tensor,
    n_iters: int,
    u_init: Optional[nn.Parameter] = None,
    v_init: Optional[nn.Parameter] = None,
) -> tuple:
    if u_init is None:
        u_init = nn.Parameter(torch.randn(W.shape[0]), requires_grad=False)
    if v_init is None:
        v_init = nn.Parameter(torch.randn(W.shape[1]), requires_grad=False)

    W_T = W.transpose(0, 1)

    for _ in range(n_iters):
        v_init = torch.matmul(W_T, u_init)
        v_init = v_init / torch.norm(v_init, p=2)
        u_init = torch.mv(W, v_init)
        u_init = u_init / torch.norm(u_init, p=2)

    # sigma = torch.sum(u_init * torch.mv(W, v_init))
    sigma = torch.matmul(torch.matmul(u_init.T, W), v_init).item()

    return sigma, u_init, v_init


def test_power_iteration_method():
    print("Start testing power iteration method")
    W = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float32)
    U, S, V = np.linalg.svd(W)
    W_tensor = torch.tensor(W)
    sigma, u, v = power_iteration_method(W_tensor, n_iters=10)
    print(np.allclose(S[0], sigma))
    print(np.allclose(u, U[:, 0]))
    print(np.allclose(v, V[0, :]))


class USPS(torch.utils.data.Dataset):

    def __init__(
        self,
        train: bool = True,
        spat_dim=(16, 16),
        download: bool = False,
    ) -> None:
        TRANSFORM = TVtransforms.Compose(
            [
                TVtransforms.Resize(spat_dim),
                TVtransforms.ToTensor(),
                TVtransforms.Normalize([0.5], [0.5]),
            ]
        )
        self.usps = TVdatasets.USPS(
            root="./data", train=train, download=download, transform=TRANSFORM
        )

    def __len__(self):
        return len(self.usps)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.usps[idx]


class SpectralNormConv2D(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.power_iterations = kwargs.pop("power_iterations")
        super().__init__(*args, **kwargs)
        self.u = nn.Parameter(torch.randn(self.weight.shape[0]), requires_grad=False)
        self.v = nn.Parameter(torch.randn(self.weight.shape[1]), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        W = self.weight.view(self.weight.shape[0], -1)
        with torch.no_grad():
            sigma, u, v = power_iteration_method(
                W, n_iters=self.power_iterations, u_init=self.u, v_init=self.v
            )

        self.u.data = u.data
        self.v.data = v.data
        self.weight.data = self.weight.data / sigma

        return super().forward(input)


class SpectralNormLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.power_iterations = kwargs.pop("power_iterations")
        super().__init__(*args, **kwargs)

        self.u = nn.Parameter(torch.randn(self.weight.shape[0]), requires_grad=False)
        self.v = nn.Parameter(torch.randn(self.weight.shape[1]), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        W = self.weight.view(self.weight.shape[0], -1)
        with torch.no_grad():
            sigma, u, v = power_iteration_method(
                W, n_iters=self.power_iterations, u_init=self.u, v_init=self.v
            )

        self.u.data = u.data
        self.v.data = v.data
        self.weight.data = self.weight.data / sigma

        return super().forward(input)


class ConvGenerator(nn.Module):

    def __init__(self, input_size: int = 128, use_batch_norm: bool = True) -> None:
        super().__init__()
        self.n_channels = 64
        self.input_size = input_size
        self.linear_block = nn.Sequential(
            nn.Linear(input_size, 4 * 4 * 4 * self.n_channels),
            nn.ReLU(True),
        )
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(4 * self.n_channels, 2 * self.n_channels, 2, stride=2),
            nn.BatchNorm2d(2 * self.n_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * self.n_channels, self.n_channels, 2, stride=2),
            nn.BatchNorm2d(self.n_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(self.n_channels, 1, 3, padding=1),
        )
        self.noise = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.size(1) == self.input_size
        output = self.linear_block(input)
        output = output.view(-1, 4 * self.n_channels, 4, 4)
        output = self.conv_block(output)
        output = torch.tanh(output)
        return output.view(-1, 1, 16, 16)

    def sample(self, n_samples: int) -> torch.Tensor:
        z = self.noise.sample([n_samples, self.input_size]).to(
            next(iter(self.parameters()))
        )
        return self.forward(z)


class ConvDiscriminator(nn.Module):

    def __init__(self, power_iterations=2) -> None:
        super().__init__()
        self.n_channels = 64
        self.power_iterations = power_iterations

        self.net = nn.Sequential(
            SpectralNormConv2D(
                1,
                self.n_channels,
                3,
                stride=2,
                padding=1,
                power_iterations=power_iterations,
            ),
            nn.LeakyReLU(),
            SpectralNormConv2D(
                self.n_channels,
                2 * self.n_channels,
                3,
                stride=2,
                padding=1,
                power_iterations=power_iterations,
            ),
            nn.LeakyReLU(),
            SpectralNormConv2D(
                2 * self.n_channels,
                4 * self.n_channels,
                3,
                stride=2,
                padding=1,
                power_iterations=power_iterations,
            ),
            nn.LeakyReLU(),
        )
        self.linear = SpectralNormLinear(
            4 * 2 * 2 * self.n_channels, 1, power_iterations=power_iterations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        output = output.view(-1, 4 * 2 * 2 * self.n_channels)
        output = self.linear(output)
        return output


def w_gen_step(
    X: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
) -> torch.Tensor:
    G.train()
    D.eval()
    batch_size = X.size(0)
    X_gen = G.sample(batch_size)
    scores_gen = D(X_gen)
    loss = -scores_gen.mean()
    G_optim.zero_grad()
    loss.backward()
    G_optim.step()
    return loss.item()


def w_discr_step(
    X: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    D_optim: torch.optim.Optimizer,
) -> torch.Tensor:
    G.eval()
    D.train()
    batch_size = X.size(0)
    with torch.no_grad():
        X_gen = G.sample(batch_size)
    scores_gen = D(X_gen)
    scores_real = D(X)
    loss = scores_gen.mean() - scores_real.mean()

    D_optim.zero_grad()
    loss.backward()
    D_optim.step()
    return loss.item()


def train_wgan(
    train_loader: DataLoader,
    G: nn.Module,
    D: nn.Module,
    G_optim: torch.optim.Optimizer,
    D_optim: torch.optim.Optimizer,
    critic_steps: int,
    n_epochs: int,
    # c: torch.Tensor = torch.tensor(0.01).to(DEVICE),
    visualize_steps: int = 10,
) -> None:

    G.train()
    D.train()
    step_i = 0
    for epoch_i in tqdm(range(n_epochs)):
        for batch_i, X in enumerate(train_loader):
            X, _ = X
            X = X.to(DEVICE)

            d_loss = w_discr_step(X, G, D, D_optim)
            wandb.log(
                {
                    "discriminator_loss": d_loss,
                }
            )

            if step_i % critic_steps == 0:
                g_loss = w_gen_step(X, G, D, G_optim)
                wandb.log(
                    {
                        "generator_loss": g_loss,
                    }
                )

            step_i += 1

        if visualize_steps and epoch_i % visualize_steps == 0:
            G.eval()
            result = G.sample(10).detach().cpu().numpy()
            plot_images(
                (result, [f"ex {idx}" for idx in range(1, 11, 1)]),
                f"media/{EXP_NAME}.png",
            )


def main():
    test_power_iteration_method()

    usps = USPS(train=True, download=True)
    plot_images(next(iter(DataLoader(usps, batch_size=10))), "media/usps_example.png")

    wandb.init(project="SMILES-GANs", name=EXP_NAME)

    train_mnist_loader = DataLoader(usps, batch_size=BATCH_SIZE, shuffle=True)
    G = ConvGenerator(use_batch_norm=False).to(DEVICE)
    D = ConvDiscriminator(power_iterations=POWER_IT).to(DEVICE)
    G_optim = torch.optim.RMSprop(G.parameters(), lr=LR)
    D_optim = torch.optim.RMSprop(D.parameters(), lr=LR)

    train_wgan(
        train_mnist_loader,
        G,
        D,
        G_optim,
        D_optim,
        critic_steps=CRITIC_STEPS,
        n_epochs=N_EPOCHS,
        visualize_steps=1,
    )


if __name__ == "__main__":
    main()
