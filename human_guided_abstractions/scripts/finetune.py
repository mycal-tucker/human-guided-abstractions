import csv
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import human_guided_abstractions.settings as settings
from human_guided_abstractions.data_utils.mnist import get_fashion_classification
from human_guided_abstractions.models.encoders import MNISTEnc
from human_guided_abstractions.models.finetune_head import FinetuneHead


def evaluate(predictor, eval_dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_dataset):
            encs, labels = batch
            pred = predictor(encs)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def get_cached_encs_dataset(dataset, encoder, label_type):
    encs = []
    all_labels = []
    for i, batch in enumerate(dataset):
        if finetune_crude:
            inputs, labels, _, _ = batch
        else:
            inputs, _, labels, _ = batch
        inputs = inputs.to(settings.device)
        labels = labels.to(settings.device)
        if label_type is not None:
            labels = labels.to(label_type)
        with torch.no_grad():
            enc, _, _ = encoder(inputs)
        # If you want to make sure it's not just initialization effects, shift all encodings.
        enc = enc * 5
        enc += 3
        encs.append(enc)
        all_labels.append(labels)
    encs = torch.vstack(encs)
    all_labels = torch.hstack(all_labels)
    enc_dataset = torch.utils.data.TensorDataset(encs, all_labels)
    enc_dataloader = torch.utils.data.DataLoader(enc_dataset, batch_size=batch_size, shuffle=True)
    return enc_dataloader


def train(encoder, predictor, train_data, test_data, label_type, criterion):
    optimizer = optim.Adam(predictor.parameters(), lr=start_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=False)
    train_data = get_cached_encs_dataset(train_data, encoder, label_type)
    test_encs = get_cached_encs_dataset(test_data, encoder, label_type)
    for epoch in tqdm(range(num_epochs)):
        for i, batch in enumerate(train_data):
            if cache_encs:
                enc, labels = batch
            else:
                if finetune_crude:
                    inputs, labels, _, _ = batch
                else:
                    inputs, _, labels, _ = batch
                inputs = inputs.to(settings.device)
                with torch.no_grad():
                    enc, _, _ = encoder(inputs)
            labels = labels.to(settings.device)
            pred = predictor(enc)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        if lr < 0.0000001:
            break
    acc = evaluate(predictor, test_encs)
    return acc


def run_trial(seed):
    loadpath = 'figures/' + load_task + '/' + enc_type + tok_suffix + '/seed' + str(load_seed) + '/'
    accs_by_num_data = []
    cached_mses = []
    cached_epochs = []
    for factor_idx, data_factor in enumerate(finetune_data_factors):
        print("******* For finetuning data factor ", data_factor, '********')
        train_data, test_data, fine_dim, crude_dim, task_criterion, label_type, _ = dataset_fn(batch_size, data_factor, choosing_idx=choosing_idx)
        pred_dim = crude_dim if finetune_crude else fine_dim
        accs = []
        mses = []
        with open(loadpath + 'mses_epoch' + str(last_epoch) + '.pkl', 'rb') as f:
            all_mses = pickle.load(f)
        curr_epoch = start_epoch
        prev_mse = None

        while curr_epoch < last_epoch:
            # Check if the MSE has increased by enough.
            new_mse = all_mses[curr_epoch]
            if len(set_epochs) > 0 and curr_epoch not in set_epochs:
                curr_epoch += 1
                continue
            if prev_mse is not None and new_mse - prev_mse < mse_step_size and curr_epoch not in set_epochs:
                curr_epoch += 1
                continue
            # 1) Load the pretrained model. We keep the encoder part, but we'll throw in a new predictor
            team = torch.load(loadpath + 'team_epoch' + str(curr_epoch) + '.pt')
            team.eval()
            team.to(settings.device)
            prev_mse = new_mse
            # 2) Create the new linear predictor.
            predictor = FinetuneHead(team.encoder.output_dim, pred_dim, num_layers=num_layers)
            predictor.to(settings.device)

            test_acc = train(team.encoder, predictor, train_data, test_data, label_type, task_criterion)
            accs.append(test_acc)
            mses.append(new_mse)
            if factor_idx == 0:
                cached_epochs.append(curr_epoch)

            curr_epoch += 1
        accs_by_num_data.append(accs)
        cached_mses.append(mses)

    # Write the results to a file so we can do average across trials or something after.
    # Format is list of lists where each row corresponds to an epoch and associated metrics.
    # Columns are:
    # 1) Random seed (set in this loop)
    # 2) Epoch number
    # 3) MSE from training
    # 5) Column for each number of finetuned data.
    print(cached_epochs)
    print(cached_mses)
    print(accs_by_num_data)
    headers = ['Seed', 'Epoch', 'MSE'] + finetune_data_factors
    write_data = [headers]
    for idx in range(len(cached_epochs)):
        row_data = [seed, cached_epochs[idx], cached_mses[0][idx]] + [entry[idx] for entry in accs_by_num_data]
        write_data.append(row_data)
    # Actually write to a file.
    epoch_time = int(time.time())
    with open(loadpath + 'finetune_results_' + str(task_suffix) + str(crude_dim) + '_' + str(seed) + '_' + str(epoch_time) + '.csv', 'w') as f:
        writer = csv.writer(f)
        for row in write_data:
            writer.writerow(row)
    print("Results for datafactors", finetune_data_factors)
    print("Epochs", cached_epochs)
    print("MSEs", cached_mses)
    print("Test Accs", accs_by_num_data)
    print()


def run():
    for seed in seeds:
        print("******* Running finetuning for load seed", load_seed, "finetune seed", seed, "**********\n")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        settings.seed = seed
        run_trial(seed)


if __name__ == '__main__':
    plt.switch_backend('agg')
    pd.set_option('display.max_columns', None)

    # What random seeds to use when running finetuning analysis.
    # It's really important to use a large number of random seeds (e.g., 10) to get an accurate estimate of finetuning
    # performance.
    seeds = [i for i in range(10)]

    finetune_crude = True
    task_suffix = 'crude' if finetune_crude else 'fine'
    choosing_idx = 0 if finetune_crude else 1
    cache_encs = True

    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 100
    batch_size = 256
    num_layers = 4
    start_lr = 0.00001

    # How many repeats of each class do you want to use (k in the paper)
    finetune_data_factors = [1, 2, 5, 10, 50]
    task = 'fashion_mnist'

    # These parameters set which models from which epochs to load
    load_task = 'fashion_mnist'
    enc_type = 'VQVIBC'
    num_tok = 1
    tok_suffix = '_n' + str(num_tok) if 'VQVIB' in enc_type else ''
    for load_seed in range(0, 5):
        start_epoch = 40
        last_epoch = 199
        mse_step_size = 0.001

        set_epochs = []
        dataset_fn = get_fashion_classification
        feature_extractor_class = MNISTEnc
        run()
