import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

LINE_STYLES = ["dashed", "dashdot", "solid", "dotted"]
COLORS = ["chocolate", "g", "tab:blue", "firebrick", "tab:red"]


def create_metric_plot(checkpoints, args):
    per_exp_train_tracked_values = [checkpoint["train_evaluator"] for checkpoint in checkpoints]
    per_exp_val_tracked_values = [checkpoint["val_evaluator"] for checkpoint in checkpoints]

    fig, ax = plt.subplots(1, figsize=(3, 2.7))

    __populate_plot(ax, per_exp_train_tracked_values, per_exp_val_tracked_values, args)

    plt.tight_layout()
    __set_size(2.25, 1.7, ax=ax)
    if args.save_plot_to:
        plt.savefig(args.save_plot_to, dpi=250, bbox_inches='tight', pad_inches=0.1)

    plt.show()


def __set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def __populate_plot(ax, per_exp_train_tracked_values, per_exp_val_tracked_values, args):
    per_exp_iterations_seqs, per_exp_metric_values_seqs = __extract_plot_info_from_evaluators(per_exp_train_tracked_values,
                                                                                              per_exp_val_tracked_values,
                                                                                              args)
    if len(args.experiments_checkpoint_paths) > 1:
        __populate_multiple_exps_single_metric_plot(ax, per_exp_iterations_seqs, per_exp_metric_values_seqs, args)
    else:
        __populate_single_exp_metric_plot(ax, per_exp_iterations_seqs[0], per_exp_metric_values_seqs[0], args)

    ax.set_title(args.plot_title, fontsize=12, pad=10)
    ax.set_ylabel(args.y_label, fontsize=11)
    ax.set_xlabel("iterations", fontsize=11)
    ax.autoscale(enable=True, axis='x', tight=True)
    if args.y_bottom_lim is not None:
        ax.set_ylim(bottom=args.y_bottom_lim)

    ax.tick_params(labelsize=9)


def __populate_multiple_exps_single_metric_plot(ax, per_exp_iterations_seqs, per_exp_metric_values_seqs, args):
    per_exp_iterations = [iterations_seqs[0] for iterations_seqs in per_exp_iterations_seqs]
    per_exp_metric_values = [metric_values_seqs[0] for metric_values_seqs in per_exp_metric_values_seqs]

    for i, (iterations, metric_values) in enumerate(zip(per_exp_iterations, per_exp_metric_values)):
        ax.plot(iterations, metric_values, color=COLORS[i], label=args.per_experiment_label[i],
                linestyle=LINE_STYLES[i], linewidth=args.plot_linewidth)

    ax.legend()


def __populate_single_exp_metric_plot(ax, iterations_seqs, metric_values_seqs, args):
    if len(metric_values_seqs) == 1:
        ax.plot(iterations_seqs[0], metric_values_seqs[0], color=COLORS[0], linewidth=args.plot_linewidth)
    else:
        for j, metric_values in enumerate(metric_values_seqs):
            color_scale = j / 8 if j < 5 else 0.5 + (j - 4) / 20
            ax.plot(iterations_seqs[j], metric_values, color=plt.cm.summer(color_scale), linewidth=args.plot_linewidth)


def __extract_plot_info_from_evaluators(per_exp_train_tracked_values, per_exp_val_tracked_values, args):
    per_exp_iterations_seqs, per_exp_metric_values_seqs = [], []

    for train_tracked_values, val_tracked_values in zip(per_exp_train_tracked_values, per_exp_val_tracked_values):
        loss_tracked_value = train_tracked_values[args.loss_metric_name]
        loss_epochs = loss_tracked_value["epochs_with_values"]
        loss_values = loss_tracked_value["epoch_values"]
        epoch_after_min_loss = __get_epoch_after_reaching_min_loss(loss_epochs, loss_values, min_loss=args.min_loss)

        metric_epochs_seqs = []
        metric_values_seqs = []

        for metric_name in args.metric_names:
            metric_tracked_value = val_tracked_values[metric_name] if metric_name in val_tracked_values else train_tracked_values[metric_name]
            metric_epochs = metric_tracked_value["epochs_with_values"]
            metric_values = metric_tracked_value["epoch_values"]

            metric_epochs, metric_values = __truncate_after_min_loss_or_max_iter(metric_epochs, metric_values, epoch_after_min_loss,
                                                                                 max_iter=args.max_iter)
            metric_epochs_seqs.append(metric_epochs)
            metric_values_seqs.append(metric_values)

        per_exp_iterations_seqs.append(metric_epochs_seqs)
        per_exp_metric_values_seqs.append(metric_values_seqs)

    return per_exp_iterations_seqs, per_exp_metric_values_seqs


def __get_epoch_after_reaching_min_loss(loss_epochs, loss_values, min_loss: float = 1e-6):
    if min_loss < 0:
        return loss_epochs[-1]

    for index, loss_value in enumerate(loss_values):
        if loss_value < min_loss:
            return loss_epochs[index]

    return loss_epochs[-1]


def __truncate_after_min_loss_or_max_iter(metric_epochs, metric_values, epoch_after_min_loss, max_iter: int = -1):
    np_epochs = np.array(metric_epochs)
    truncate_at = epoch_after_min_loss if max_iter < 0 else min(epoch_after_min_loss, max_iter)

    num_before_max_iter = np_epochs.searchsorted(truncate_at, side="right")
    return metric_epochs[: num_before_max_iter], metric_values[: num_before_max_iter]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_checkpoint_paths", nargs="+", type=str, required=True, help="Path to the experiments checkpoints.")

    p.add_argument("--per_experiment_label", nargs="+", type=str, default=[], help="Label for each experiment. Supported only when plotting a single "
                                                                                   "metric over multiple experiments.")
    p.add_argument("--metric_names", nargs="+", type=str, default=["mse_loss"], help="Name of metrics to plot. If multiple experiments are given "
                                                                                     "will take only the first metric (multiple metrics for "
                                                                                     "multiple experiments is not supported).")
    p.add_argument("--max_iter", type=int, default=-1, help="Maximal number of iterations to plot. If -1 will not truncate according to iterations.")
    p.add_argument("--min_loss", type=float, default=-1, help="Minimal loss value to plot until.")
    p.add_argument("--loss_metric_name", type=str, default="mse_loss", help="Name of the loss metric.")
    p.add_argument("--plot_title", type=str, default="", help="Title for the plot.")
    p.add_argument("--y_label", type=str, default="mean square error", help="Name of metric to plot.")
    p.add_argument("--y_bottom_lim", type=float, default=None, help="Bottom limit for y axis")
    p.add_argument("--plot_linewidth", type=float, default=1.5, help="Plots line width.")
    p.add_argument("--save_plot_to", type=str, default="", help="Save plot to the given file path (doesn't save if non given)")
    args = p.parse_args()

    checkpoints = [torch.load(checkpoint_path, map_location=torch.device("cpu")) for checkpoint_path in args.experiments_checkpoint_paths]
    create_metric_plot(checkpoints, args)


if __name__ == "__main__":
    main()
