
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torchvision.transforms as transforms
import torch

from sklearn.decomposition import PCA


# Plots MNIST inputs (optionally) and reconstructions.
def viz_mnist(recons, originals=None, savepath=None):
    num_imgs = recons.shape[0]
    max_num = 30
    if num_imgs > max_num:
        recons = recons[:max_num]
        if originals is not None:
            originals = originals[:max_num]
        num_imgs = recons.shape[0]
    reshaped_recons = np.reshape(recons, (num_imgs, 28, 28))
    if originals is not None:
        num_rows = 2
        reshaped_og = np.reshape(originals, (num_imgs, 28, 28))
    else:
        num_rows = 1
    _, axes = plt.subplots(num_rows, num_imgs, figsize=(20, 3))
    if num_rows == 1:
        axes = [axes]
    for i in range(num_imgs):
        recons_ax = axes[-1][i]
        recons_ax.imshow(reshaped_recons[i], cmap='gray_r')
        recons_ax.axis('off')
        if originals is not None:
            og_ax = axes[0][i]
            og_ax.imshow(reshaped_og[i], cmap='gray_r')
            og_ax.axis('off')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
        return
    plt.show()


# Plots MNIST reconstructions (top row) and a histogram (bottom row). Useful for visualizing learned VQ vectors and
# their relative frequencies.
def viz_vq(recons, likelihoods, img_type, savepath=None, savepath_suffix=None):
    num_imgs = min([len(recons), 30])  # Set whatever cutoff you want
    recons = recons[:num_imgs]
    likelihoods = likelihoods[:num_imgs]
    if img_type == 'mnist':
        reshaped_recons = np.reshape(recons, (num_imgs, 28, 28))
    elif img_type == 'cifar':
        recons = recons / 2 + 0.5  # Unnormalize because we rescaled to [-1, 1]
        reshaped_recons = recons
    elif img_type == 'inat':
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        reshaped_recons = inv_normalize(torch.Tensor(recons)).numpy()
    elif img_type is None:
        reshaped_recons = [None for _ in range(num_imgs)]
    else:
        assert False, "Bad img type " + img_type
    # First, just save all the prototypes as their own files.
    if img_type != 'mnist':
        for idx, img in enumerate(reshaped_recons):
            if img is None:
                continue
            single_img = np.transpose(img, (1, 2, 0))
            ax = plt.subplot(1, 1, 1)
            ax.imshow(single_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(savepath + '_' + savepath_suffix + '_proto' + str(idx), bbox_inches='tight')
            plt.close()
    plt.figure(figsize=(12, 6))
    for i in range(num_imgs):
        ax = plt.subplot(2, num_imgs, i + 1)
        if img_type == 'mnist':
            ax.imshow(reshaped_recons[i], cmap='gray_r')
        elif img_type == 'cifar' or img_type == 'inat':
            single_img = reshaped_recons[i]
            single_img = np.transpose(single_img, (1, 2, 0))
            ax.imshow(single_img)

        ax.axis('off')
    # Put the histogram below
    ax = plt.subplot(2, 1, 2)
    ax.bar([i for i in range(num_imgs)], likelihoods, width=0.8)
    ax.margins(x=0)  # Remove margins so that the bars line up with the images above them.
    ylim = max(0.11, np.max(likelihoods) + 0.01)
    ax.set_ylim(0, ylim)
    if savepath is not None:
        plt.savefig(savepath)
        plt.savefig(savepath + '_' + savepath_suffix)
        plt.close()
        return
    plt.show()


# Plot 2D PCA of the communication vectors, with optional decorations like the reconstructed images.
# Currently assumes MNIST, which, again, is not great...
def plot_pca(comms, extra_points=None, imgs=None, sizes=None, coloring=None, savepath=None, savepath_suffix=None, img_type='mnist'):
    if len(comms) == 1:
        print("Only one comms; returning without trying to do PCA")
        return
    _, ax = plt.subplots()
    pca = PCA(n_components=2)
    pca.fit(comms)

    if extra_points is not None:
        transformed = pca.transform(extra_points)
        pcm = ax.scatter(transformed[:, 0], transformed[:, 1], s=0.5, c='k', alpha=0.1)

    if comms.shape[1] > 2:
        transformed = pca.transform(comms)
    else:
        transformed = comms

    s = 200 if sizes is None else sizes
    # We don't actually see these points right now because they're covered by the images, but if the images aren't
    # passed in, you would see them.
    pcm = ax.scatter(transformed[:, 0], transformed[:, 1], s=s, c=coloring, marker='o')

    # And put the images in the scatter plot too.
    if imgs is not None:
        num_imgs = min([len(imgs), 20])  # Set whatever cutoff you want
        if img_type == 'mnist':
            recons = np.reshape(imgs[:num_imgs], (num_imgs, 28, 28))  # Only works for MNIST.
        elif img_type == 'inat':
            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            recons = inv_normalize(torch.Tensor(imgs[:num_imgs])).numpy()
        elif img_type == 'cifar':
            recons = imgs[:num_imgs]
            recons = recons / 2 + 0.5  # Unnormalize because we rescaled to [-1, 1]
        for i, img in enumerate(recons):
            if img_type == 'mnist':
                imagebox = OffsetImage(img, zoom=sizes[i] / 100, cmap='gray_r')
            elif img_type == 'inat':
                single_img = np.transpose(img, (1, 2, 0))
                # iNat images are 224 by 224, so need to divide size by about 10x vs. MNIST
                imagebox = OffsetImage(single_img, zoom=sizes[i] / 1000)
            elif img_type == 'cifar':
                single_img = np.transpose(img, (1, 2, 0))
                imagebox = OffsetImage(single_img, zoom=sizes[i] / 100)
            ab = AnnotationBbox(imagebox, (transformed[i, 0], transformed[i, 1]), frameon=False)
            ax.add_artist(ab)
            # Save locations of proto transforms in a file.
            with open(savepath + '_' + savepath_suffix + '_locs.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(transformed[i])
    if savepath is not None:
        plt.savefig(savepath)
        plt.savefig(savepath + '_' + savepath_suffix)
        plt.close()
        return
    plt.show()


def plot_scatter(x_values, y_values, savepath, savepath_suffix, xlabel, ylabel):
    plt.scatter(x_values, y_values, c=[i for i in range(len(x_values))])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.savefig(savepath + '_' + savepath_suffix)
    plt.close()



def plot_multi_scatter(x_values, y_values, y_errors, savepath, xlabel, ylabel, labels, title=''):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Use the viridis colormap to get five different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_values) + 1))

    idx = 0
    for x_vals, y_vals, label in zip(x_values, y_values, labels):
        color = colors[idx]
        plt.scatter(x_vals, y_vals, label=label, alpha=0.2, color=color)
        good_idxs = np.isfinite(x_vals) & np.isfinite(y_vals)  # Remove nans
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)
        x_vals = x_vals[good_idxs]
        y_vals = y_vals[good_idxs]
        poly = np.polyfit(x_vals[good_idxs], y_vals[good_idxs], deg=3)
        plot_x = np.arange(min(x_vals), max(x_vals), 0.001)
        plt.plot(plot_x, np.polyval(poly, plot_x), color=color)
        idx += 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    legend = plt.legend()
    for lh in legend.legendHandles:
        lh.set_facecolor(lh.get_facecolor())
        lh.set_edgecolor(lh.get_edgecolor())
        lh.set_alpha(1.0)
    plt.ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(savepath + '.png')
    plt.close()


def plot_trend(y_values, savepath, savepath_suffix, xlabel, ylabel, x_values=None, legend=None):
    if isinstance(y_values[0], list):  # Just plot multiple trends.
        num_trends = len(y_values[0])
        for i in range(num_trends):
            y_vals = [elt[i] for elt in y_values]
            if x_values is None:
                plt.plot(y_vals)
            else:
                plt.plot(x_values, y_values)
    else:
        if x_values is None:
            plt.plot(y_values)
        else:
            plt.plot(x_values, y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        legend = plt.legend(legend)
        for lh in legend.legendHandles:
            lh.set_facecolor(lh.get_facecolor())
            lh.set_edgecolor(lh.get_edgecolor())
            lh.set_alpha(1.0)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.savefig(savepath + '_' + savepath_suffix)
    plt.close()
