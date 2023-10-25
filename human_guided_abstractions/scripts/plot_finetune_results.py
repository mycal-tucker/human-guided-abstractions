
import pandas as pd
import matplotlib.pyplot as plt
import glob

from human_guided_abstractions.utils.plotting import plot_multi_scatter


def get_finetune_stats(enc_type, load_seed):
    # Load the data across random finetuning seeds for that particular setup. Return mean, std.
    loadpath = 'figures/' + load_task + '/' + enc_type + '/seed' + str(load_seed) + '/'
    dfs = []
    for seed in finetune_seeds:
        base = loadpath + 'finetune_results_' + task_suffix + '_' + str(seed)
        files = glob.glob(base + '*.csv')
        for file in files:
            print("Loading file", file)
            df = pd.read_csv(file)
            dfs.append(df)
    catted = pd.concat(dfs)
    print("Overall results\n", catted.head())
    grouped = catted.groupby('Epoch')
    return grouped.mean(), grouped.std()


def run_for_df(data_factor):
    all_mses = []
    all_accs = []
    all_stds = []
    labels = []
    for enc_type in enc_types:
        enc_means = []
        enc_stds = []
        for load_seed in load_seeds:
            run_means, run_stds = get_finetune_stats(enc_type, load_seed)
            enc_means.append(run_means)
            enc_stds.append(run_stds)
        catted_means = pd.concat(enc_means)
        print("Overall mean\n")
        print(catted_means.mean(axis=1))
        catted_stds = pd.concat(enc_stds)
        mses = catted_means['MSE']
        accs = catted_means[str(data_factor)]
        all_mses.append(mses)
        all_accs.append(accs)
        all_stds.append(catted_stds[str(data_factor)])
        # Write a CSV with the data from all seeds for this encoder?
        loadpath = 'figures/' + load_task + '/' + enc_type + '/'
        catted_means.to_csv(loadpath + enc_type + '_all_data_' + str(num_crude) + '.csv')
        catted_stds.to_csv(loadpath + enc_type + '_all_stds_' + str(num_crude) + '.csv')
        if 'VAE' in enc_type:
            pretty_enc_name = r'$\beta$-VAE $k=' + str(data_factor) + '$'
        elif 'VQVIBC' in enc_type:
            pretty_enc_name = 'VQ-VIB$_\mathcal{C}$' + ' $k=' + str(data_factor) + '$'
        elif 'VQVIBN' in enc_type:
            pretty_enc_name = 'VQ-VIB$_\mathcal{N}$' + ' $k=' + str(data_factor) + '$'
        else:
            assert False
        labels.append(pretty_enc_name)

    title = "FashionMNIST " + str(num_crude) + "-Way Accuracy; $k = " + str(data_factor) + '$'
    plot_multi_scatter(all_mses, all_accs, all_stds, 'figures/' + load_task + '/multi_' + task_suffix + '_' + str(data_factor), 'MSE', 'Y Acc.', labels, title=title)
    return all_accs, all_mses, all_stds, labels


def run():
    all_accs = []
    all_mses = []
    all_stds = []
    all_labels = []
    for data_factor in data_factors:
        accs, mses, stds, labels = run_for_df(data_factor)
        all_accs.extend(accs)
        all_mses.extend(mses)
        all_stds.extend(stds)
        all_labels.extend(labels)
    title = 'FashionMNIST 3-Way Accuracy'
    plot_multi_scatter(all_mses, all_accs, all_stds,
                       'figures/' + load_task + '/multi_' + task_suffix + '_all', 'MSE', 'Y Acc.',
                       all_labels, title=title)


if __name__ == '__main__':
    plt.switch_backend('agg')

    finetune_seeds = [i for i in range(10)]
    finetune_crude = True
    num_crude = 3
    task_suffix = 'crude' if finetune_crude else 'fine'
    task_suffix += str(num_crude)

    load_task = 'fashion_mnist'
    enc_types = ['VQVIBC_n1']
    load_seeds = [0, 1, 2, 3, 4]

    data_factors = [1, 2, 5, 10, 50]
    run()

