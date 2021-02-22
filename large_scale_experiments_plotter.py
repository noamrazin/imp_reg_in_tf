import argparse
import glob
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

SCALE_FACTOR = 4

RAND_LABELS_PLOT_MARKER = 'd'

RAND_DATA_PLOT_MARKER = '^'

NATURAL_PLOT_MARKER = 'o'

VARIANCE_PLOT_COLOR = 'tab:green'

TITLE_SIZE = 12

LABELS_SIZE = 11

TICKS_SIZE = 9

RAND_LABELS_PLOT_COLOR = 'tab:purple'

RAND_DATA_PLOT_COLOR = 'tab:red'

NATURAL_PLOT_COLOR = 'tab:blue'


def key_function(string):
    order = {'original train': 1,
             'original test': 2,
             'rand image train': 3,
             'rand image test': 4,
             'rand label train': 5,
             'rand label test': 6,
             'variance': 7}
    if string in order:
        return order[string]
    return 8


def get_plot_style_dict():
    styles = {'natural': dict(color=NATURAL_PLOT_COLOR, marker=NATURAL_PLOT_MARKER),
              'rnd_lbl': dict(color=RAND_LABELS_PLOT_COLOR, marker=RAND_LABELS_PLOT_MARKER),
              'rnd_data': dict(color=RAND_DATA_PLOT_COLOR, marker=RAND_DATA_PLOT_MARKER),
              'train': dict(linestyle='-', alpha=1.0),
              'test': dict(linestyle='--', alpha=0.5)}
    plot_labels = {'natural': 'original', 'rnd_data': 'rand image', 'rnd_lbl': 'rand label'}
    plot_titles = {'mnist': 'MNIST', 'fmnist': 'Fashion-MNIST'}
    return styles, plot_labels, plot_titles


def fetch_data(root_directory_path):
    checkpoints = glob.glob(os.path.join(root_directory_path, '*/*/*/checkpoints/*.ckpt'))

    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for checkpoint_path in tqdm(checkpoints):
        checkpoint = torch.load(checkpoint_path)
        rank = checkpoint['model']['factors'].shape[2]
        train_loss = checkpoint['train_evaluator']['mse_loss']['current_value']
        test_loss = checkpoint['val_evaluator']['val_mse_loss']['current_value']

        dataset_type = 'other_type'
        for type_option in ['natural', 'rnd_lbl', 'rnd_data']:
            if type_option in checkpoint_path:
                dataset_type = type_option
                break

        dataset_name = 'other_dataset_name'
        for dataset_name_option in ['fmnist', 'mnist']:
            if dataset_name_option in checkpoint_path:
                dataset_name = dataset_name_option
                break

        all_data[dataset_name][(dataset_type, 'train')][rank].append(train_loss)
        all_data[dataset_name][(dataset_type, 'test')][rank].append(test_loss)

    for dataset_name, by_dataset_name_data_dict in all_data.items():
        for dataset_type, by_dataset_type_data_dict in by_dataset_name_data_dict.items():
            for rank, list_of_results in by_dataset_type_data_dict.items():
                rescale = np.array(list_of_results) / SCALE_FACTOR
                all_data[dataset_name][dataset_type][rank] = (rescale.mean(), rescale.std())

    return all_data


def populate_axes(ax, dataset_name, data):
    styles, plot_labels, plot_titles = get_plot_style_dict()

    for i, (raw_label, series) in enumerate(data.items()):
        dataset_type = raw_label[0]
        train_or_test = raw_label[1]

        x, y_values = zip(*(sorted(series.items())))
        y, yerr = zip(*y_values)

        processed_label = plot_labels[dataset_type] + ' ' + train_or_test
        ax.errorbar(x, y, yerr=yerr, label=processed_label, **styles[dataset_type], **styles[train_or_test])

    ax.plot([1, 10], [0.09, 0.09], label='variance', color=VARIANCE_PLOT_COLOR, alpha=0.5)
    ax.set_xticks(x)
    ax.tick_params(axis='x', labelsize=TICKS_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_SIZE)
    ax.set_title(plot_titles[dataset_name], fontsize=TITLE_SIZE)
    ax.set_xlim((1, 10))
    ax.set_xlabel('tensor rank', fontsize=LABELS_SIZE)
    ax.set_ylabel('mean squared error', fontsize=LABELS_SIZE)
    ax.set_yscale('log')


def create_plot(args):
    all_data = fetch_data(args.experiments_dir)

    fig, ax = plt.subplots(1, 2, figsize=(15, 3))

    handles, labels = [], []
    for axes, dataset_name in zip(ax, ['mnist', 'fmnist']):
        populate_axes(axes, dataset_name, all_data[dataset_name])
        curr_handles, curr_labels = ax[0].get_legend_handles_labels()

        for handle, label in zip(curr_handles, curr_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    handles, labels = list(zip(*sorted(list(zip(handles, labels)), key=lambda x: key_function(x[1]))))

    fig.legend(handles, labels, loc='center', ncol=1, bbox_to_anchor=(0.5, 0.5), fontsize='11')

    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.99, top=None,
                        wspace=0.76, hspace=None)

    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    if args.save_plot_to != '':
        if not os.path.exists(os.path.dirname(args.save_plot_to)):
            os.mkdir(os.path.dirname(args.save_plot_to))
        plt.savefig(args.save_plot_to + now_utc_str, dpi=250, bbox_inches='tight', pad_inches=0.01)

    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir", type=str, required=True, help="Paths to a directory with experiments.")
    p.add_argument("--save_plot_to", type=str, default="",
                   help="Save plot to the given file path (doesn't save if none given).")
    p.add_argument("--plot_title", type=str, default="", help="Title for the plot.")
    p.add_argument("--plot_scale", type=float, default=1.5, help="Determine the scale of the 4x3 plot.")
    p.add_argument("--trivial_loss_value", type=float, default=-1,
                   help="Insert trivial loss value to plot horizontal line in its place. (when left blank - no line will be plotted)")
    args = p.parse_args()

    create_plot(args)


if __name__ == '__main__':
    main()
