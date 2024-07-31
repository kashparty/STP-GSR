import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib.ticker import ScalarFormatter


def plot_grad_flow(named_parameters, step, tmp_dir):
    """Plot average and max gradient flow across layers at given step"""
    ave_grads = []
    max_grads = []
    layers = []

    # Calculate average and max gradient flow for each layer
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    
    plt.plot(ave_grads, alpha=0.3, color="b", label="Average gradient")
    plt.plot(max_grads, alpha=0.3, color="r", label="Max gradient")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Gradient Value")
    plt.title(f"Gradient flow, step {step}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot as an image file in the temporary directory
    filename = os.path.join(tmp_dir, f"grad_flow_{step:03d}.png")
    plt.savefig(filename)
    plt.close()


def create_gif_grad(image_folder, gif_name):
    """Combine gradient flow plots for individual steps into a gif"""
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.startswith('grad_flow_') and file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=500, loop=0)


def format_colorbar(cb):
    """Function to format colorbar"""
    cb.formatter = ScalarFormatter(useMathText=True)
    cb.formatter.set_scientific(False)
    cb.formatter.set_useOffset(False)
    cb.update_ticks()


def plot_adj_matrices(orig_s, orig_t, pred_t, step, tmp_dir, file_name=None):
    """Plot source, target, and predicted adjacency matrices for given step"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot source graph adjacency matrix
    cb = axs[0].imshow(orig_s, cmap='viridis')
    axs[0].set_title('Original Source')
    cb = plt.colorbar(cb, ax=axs[0])
    format_colorbar(cb)

    # Plot target graph adjacency matrix
    cb = axs[1].imshow(orig_t, cmap='viridis')
    axs[1].set_title('Original Target')
    cb = plt.colorbar(cb, ax=axs[1])
    format_colorbar(cb)

    # Plot predicted target graph adjacency matrix
    cb = axs[2].imshow(pred_t, cmap='viridis')
    axs[2].set_title('Predicted Target')
    cb = plt.colorbar(cb, ax=axs[2])
    format_colorbar(cb)
    
    plt.tight_layout()

    if file_name is None:
        file_name = f"train_samples_{step:03d}"
    
    # Save the plot as an image file in the temporary directory
    filename = os.path.join(tmp_dir, f"{file_name}.png")
    plt.savefig(filename)
    plt.close()


def create_gif_adj(image_folder, gif_name):
    """Combine adjacency matrix plots for individual steps into a gif"""
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.startswith('train_samples_') and file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=500, loop=0)


def plot_losses(losses, loss_type, res_dir):
    """Plot loss curves"""
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{loss_type} Loss')

    plt.savefig(f'{res_dir}/{loss_type}_loss.png')
    plt.close()