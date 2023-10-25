import os
import pickle
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy import stats
import shutil

import human_guided_abstractions.settings as settings
from human_guided_abstractions.data_utils.mnist import get_fashion_classification
from human_guided_abstractions.models.baselines import VAE
from human_guided_abstractions.models.decoder import MNISTDecoder
from human_guided_abstractions.models.encoders import MNISTEnc
from human_guided_abstractions.models.listener import Listener
from human_guided_abstractions.models.team import Team
from human_guided_abstractions.models.vqvibn import VQVIBN
from human_guided_abstractions.models.vqvibc import VQVIBC
from human_guided_abstractions.utils.plotting import viz_vq, plot_pca, plot_scatter, plot_trend


def evaluate(team, eval_dataset, epoch):
    print("Starting eval")
    team.eval()
    correct = 0
    total = 0
    total_loss = 0
    total_recons_loss = 0
    # Map from a quantized vector to tuple of info about it.
    # Current implementation just has a tuple of (reconstruction, labels, count)
    # One could lots of other forms of information too, of course.
    vq_to_info = {}
    raw_encs = []
    with torch.no_grad():
        for i, batch in enumerate(eval_dataset):
            if i > 100:
                print("Breaking eval after", i, "batches")
                break
            inputs, _, labels, _ = batch
            inputs = inputs.to(settings.device)
            labels = labels.to(settings.device)
            pred, _, _ = team(inputs)
            _, predicted = torch.max(pred.data, 1)
            if isinstance(task_criterion, nn.L1Loss) or isinstance(task_criterion, nn.MSELoss):
                labels = torch.unsqueeze(labels, 1)
                pred = 10 * torch.sigmoid(pred)
            pred_loss = task_criterion(pred, labels)
            total_loss += pred_weight * pred_loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            comm, raw_enc, enc_loss = team.encoder(inputs)
            if len(raw_encs) < 1000:
                raw_encs.append(raw_enc.detach().cpu().numpy())
            reconstructions = team.decoder(comm)  # For assessment, pass along the comm (never the raw encoding).
            recons_loss = recons_criterion(inputs, reconstructions).item()
            total_loss += settings.recons_weight * recons_loss + enc_loss.item()
            total_recons_loss += recons_loss * labels.size(0)  # It was already averaged by batch size.
            viz_recons = team.reconstruct(comm).cpu().numpy()
            np_labels = labels.cpu().numpy()
            precision = 2  # Round the vectors so we can find matches.
            rounded = np.around(comm.cpu().numpy(), precision)
            for j, rounded_vq in enumerate(rounded):
                tup = tuple(rounded_vq)  # Turn the numpy vector into a tuple so we can hash it.
                if tup in vq_to_info.keys():
                    prev_data = vq_to_info.get(tup)
                    vq_to_info[tup] = (prev_data[0],  prev_data[1] + [np_labels[j]], prev_data[2] + 1)
                    continue
                if team.encoder.num_tokens != -1 and len(vq_to_info.keys()) == team.encoder.num_tokens ** num_tok_per_message:
                    print("Somethings wrong")
                    print("Tup", tup)
                    print("Keys")
                    for k in vq_to_info.keys():
                        print(k)
                vq_to_info[tup] = (viz_recons[j], [np_labels[j]], 1)
    # Gather all reconstructions for the unique quantized vectors and plot them.
    comms = []
    unique_recons = []
    labels = []
    likelihoods = []
    # Iterate through all elements together to get a consistent ordering.
    for key, val in vq_to_info.items():
        comms.append(key)
        unique_recons.append(val[0])
        # There might be multiple labels associated with this vector; pick the most common one.
        # Mode returns mode value and count, but we just want value.
        labels.append(stats.mode(val[1], keepdims=False)[0])
        likelihoods.append(val[2] / total)
    comms = np.vstack(comms)
    raw_encs = np.vstack(raw_encs)
    unique_recons = np.stack(unique_recons)
    labels = np.vstack(labels)
    likelihoods = np.vstack(likelihoods)
    ent = np.sum(-1 * likelihoods * np.log2(likelihoods))
    print("Entropy (nats)", ent)
    # Sort from most to least common (use negative to get that ordering, otherwise we'd get least to most common)
    idxs = np.argsort(-likelihoods, axis=0).squeeze(1)
    unique_recons = unique_recons[idxs]
    likelihoods = likelihoods[idxs]
    likelihoods = np.squeeze(likelihoods, axis=1)
    img_type = 'mnist'
    pca_imgs = unique_recons
    if 'VQVIB' in enc_type:
        viz_vq(unique_recons, likelihoods, img_type, savepath=savepath + 'vqs', savepath_suffix=str(epoch))
    # And visualize the communication using 2D PCA. Make the point size proportional to the likelihood.
    plot_pca(comms[idxs], extra_points=raw_encs, imgs=pca_imgs, sizes=1000 * likelihoods, coloring=labels[idxs], savepath=savepath + 'pca', savepath_suffix=str(epoch), img_type=img_type)
    return correct / total, total_loss / total, total_recons_loss / total


def train(team):
    running_mse = 0
    optimizer = optim.Adam(team.parameters(), lr=0.001)
    # Track some metrics
    accuracies = []
    train_accuracies = []
    mses = []
    for epoch in range(num_epochs):
        print("Epoch", epoch, "of", num_epochs)
        team.train()
        if epoch >= burnin:
            settings.kl_weight += kl_weight_incr
            settings.entropy_weight += h_weight_incr
            settings.entropy_weight = max(0, settings.entropy_weight)
            settings.kl_weight = max(0.0, settings.kl_weight)
            print("Entropy weight", settings.entropy_weight)
            print("KL Weight", settings.kl_weight)
            settings.recons_weight += info_incr
            settings.recons_weight = max(0.0, settings.recons_weight)
            settings.decode_raw_prob = 0
        for i, batch in enumerate(train_data):
            inputs, _, train_labels, _ = batch
            optimizer.zero_grad()
            inputs = inputs.to(settings.device)
            train_labels = train_labels.to(settings.device)
            if label_type is not None:
                train_labels = train_labels.to(label_type)
            pred, recons, enc_loss = team(inputs)
            if isinstance(task_criterion, nn.L1Loss) or isinstance(task_criterion, nn.MSELoss):
                train_labels = torch.unsqueeze(train_labels, 1)
                pred = 10 * torch.sigmoid(pred)
            pred_loss = task_criterion(pred, train_labels)
            recons_loss = recons_criterion(inputs, recons)
            if enc_loss is None:
                enc_loss = torch.tensor(0)
            total_loss = pred_weight * pred_loss + settings.recons_weight * recons_loss + enc_loss
            total_loss.backward()
            optimizer.step()

            running_mse = 0.95 * running_mse + 0.05 * recons_loss.detach().cpu().item()
        test_acc, val_overall_loss, val_recons_loss = evaluate(team, test_data, epoch)
        train_acc = 0
        val_acc = 0
        print("Test accuracy", test_acc)
        print("Validation accuracy", val_acc)
        print("Train accuracy", train_acc)
        print("Recons loss", val_recons_loss)

        print("Pred loss", pred_loss.item())
        print("Enc loss", enc_loss.item())
        accuracies.append(test_acc)
        train_accuracies.append(train_acc)
        mses.append(val_recons_loss)
        plot_scatter(mses, accuracies, savepath + 'mse_acc', str(epoch), "MSE (train set)", "Accuracy (percent)")
        plot_trend(mses, savepath + 'mse', str(epoch), 'Epoch', 'MSE')
        torch.save(team, savepath + 'team_epoch' + str(epoch) + '.pt')
        # Dump some metrics into pickle files
        with open(savepath + 'mses_epoch' + str(epoch) + '.pkl', 'wb') as f:
            pickle.dump(mses, f)
        print("\n\n")


def run():
    feature_extractor = feature_extractor_class(enc_dim)
    if enc_type == 'VQVIBN':
        encoder = VQVIBN(enc_dim, num_protos=num_protos, feature_extractor=feature_extractor, num_simultaneous_tokens=num_tok_per_message).to(settings.device)
    elif enc_type == 'VQVIBC':
        encoder = VQVIBC(enc_dim, num_protos=num_protos, feature_extractor=feature_extractor, num_simultaneous_tokens=num_tok_per_message).to(settings.device)
    elif enc_type == 'VAE':
        encoder = VAE(enc_dim, feature_extractor=feature_extractor).to(settings.device)
    else:
        assert False, "Bad enc type logic for enc type " + str(enc_type)
    listener = Listener(enc_dim, pred_dim)
    team = Team(encoder, listener, decoder)
    team.to(settings.device)
    train(team)


if __name__ == '__main__':
    plt.switch_backend('agg')
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    cli_args = argp.parse_args()

    config = yaml.load(open(cli_args.experiment_config), Loader=yaml.FullLoader)
    recons_criterion = nn.MSELoss()
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weight_config = config['weights']
    settings.kl_weight = weight_config['kl_weight']
    settings.entropy_weight = weight_config['entropy_weight']
    settings.recons_weight = weight_config['recons_weight']
    pred_weight = weight_config['utility_weight']
    kl_weight_incr = weight_config['kl_incr']
    h_weight_incr = weight_config['entropy_incr']
    info_incr = weight_config['recons_incr']

    training_config = config['training']
    num_epochs = training_config['num_epochs']
    batch_size = training_config['batch_size']
    burnin = training_config['burnin']  # How many epochs to train at without annealing any weights.

    model_config = config['model']
    num_protos = model_config['num_protos']
    num_tok_per_message = model_config['num_subprotos']
    enc_dim = model_config['enc_dim']

    # Current supported tasks are:
    task = config['task']
    assert task in ['fashion_mnist']  # Add other datasets as desired
    feature_extractor_class = None
    if task == 'fashion_mnist':
        dataset_fn = get_fashion_classification
        feature_extractor_class = MNISTEnc
        decoder = MNISTDecoder(enc_dim)
    else:
        assert False, "Bad task spec"

    # Specify the type of encoder you want by setting the string enc_type.
    enc_type = model_config['class']
    allowed_encoder_types = ['VQVIBC', 'VQVIBN', 'VAE']
    assert enc_type in allowed_encoder_types, "Only allow encoder types in " + str(allowed_encoder_types)

    # Iterate over a few random seeds and configurations.
    for seed in range(0, 5):
        for num_tok_per_message in [1]:  # Specify number of quantized vectors to combine (n in the paper)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            settings.seed = seed

            weight_config = config['weights']
            settings.kl_weight = weight_config['kl_weight']
            settings.entropy_weight = weight_config['entropy_weight']
            settings.recons_weight = weight_config['recons_weight']
            pred_weight = weight_config['utility_weight']
            kl_weight_incr = weight_config['kl_incr']
            h_weight_incr = weight_config['entropy_incr']
            info_incr = weight_config['recons_incr']

            h_weight_incr = num_tok_per_message * h_weight_incr
            info_incr = num_tok_per_message * info_incr

            train_data, test_data, fine_dim, crude_dim, task_criterion, label_type, settings.og_dataset = dataset_fn(batch_size)
            pred_dim = fine_dim

            # Scalar weights for trading off losses
            tok_suffix = '_n' + str(num_tok_per_message) if 'VQVIB' in enc_type else ''
            savepath = 'figures/' + task + '/' + enc_type + tok_suffix + '/seed' + str(seed) + '/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            shutil.copyfile(cli_args.experiment_config, savepath + 'config.yaml')
            np.set_printoptions(precision=4)
            print("Training with")
            print("Seed", seed)
            print("Num epochs", num_epochs)
            print("KL Weight", settings.kl_weight)
            print("Entropy Weight", settings.entropy_weight)
            print("KL Incr", kl_weight_incr)
            print("H Incr", h_weight_incr)
            print("Recons Incr", info_incr)
            print("Burnin", burnin)
            print("Enc type", enc_type)
            print("Num protos", num_protos)
            print("Enc dim", enc_dim)
            print("Batch size", batch_size)
            print("Recons weight", settings.recons_weight)
            print("Prediction weight", pred_weight)
            print("decode_raw_prob", settings.decode_raw_prob)
            print("\n\n\n")
            run()
