import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from conditional_mnist_conv import plot_images, ConditionalConvGenerator

EXP_NAME = "ns_cond_gen_cond_disc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_examples(G: nn.Module):
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    G.eval()

    for idx in range(5):
        result = (
            G.sample(
                10, F.one_hot(targets, num_classes=10).type(torch.float).to(DEVICE)
            )
            .detach()
            .cpu()
            .numpy()
        )
        plot_images((result, targets), f"media/{EXP_NAME}/{idx}.png")


def main():
    G = ConditionalConvGenerator()
    G.load_state_dict(torch.load(f"weights/G_{EXP_NAME}.pth"))
    G.to(DEVICE)
    G.eval()
    if not os.path.exists(f"media/{EXP_NAME}"):
        os.makedirs(f"media/{EXP_NAME}")
    sample_examples(G)


if __name__ == "__main__":
    main()
