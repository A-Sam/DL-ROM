from utils import plot_training_from_dict
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from dataset_create import createAnimation

"""
python visualize.py -mode results -d_set 2d_cylinder -freq 10
python visualize.py -mode simulate -d_set 2d_cylinder
"""


def preprocess_plot_data(arr):
    return abs(arr)


def plot_results(pred, labels, mode, dataset_name, freq):
    assert pred.shape == labels.shape

    for i in range(0, pred.shape[0], freq):
        print(f"Plotted {i} / {pred.shape[0]}")
        plt.imshow(preprocess_plot_data(pred[i]), cmap="turbo", vmin=0, vmax=0.5)
        # fig, ax = plt.subplots(figsize=(3, 1))
        # plt.colorbar(plt.cm.ScalarMappable( cmap='turbo'),
        #            ax=ax,
        #            orientation='vertical',
        #            ticks=[-1,
        #                   1],
        #            pad=0.01)
        # plt.axis("off")
        plt.savefig(f"../{mode}/{dataset_name}/plots/pred_{i}.png", dpi=600)
        plt.close()
        # print("pred[" + str(i) + "]" + str(pred[i]))

        plt.imshow(preprocess_plot_data(labels[i]), cmap="turbo", vmin=0, vmax=0.5)
        # plt.axis("off")
        plt.savefig(f"../{mode}/{dataset_name}/plots/label_{i}.png", dpi=600)
        plt.close()
        # print("labels[" + str(i) + "]" + str(labels[i]))

        pred_label_diff = preprocess_plot_data(pred[i] - labels[i])
        plt.imshow(pred_label_diff, cmap="turbo", vmin=0, vmax=0.5)
        # plt.axis('off')
        plt.savefig(f"../{mode}/{dataset_name}/plots/diff_{i}.png", dpi=600)
        plt.close()
        # print("pred_label_diff[" + str(i) + "]" + str(pred_label_diff))


def plot_simulate(mse):
    plt.plot(-1 * np.log(mse), "ko-")
    plt.xlabel("Epoch Number")
    plt.ylabel("Negative Log. MSE (per pixel)")

    plt.savefig(
        f"../{mode}/{dataset_name}/plots/mse_lineplot.png", bbox_inches="tight", dpi=600
    )
    plt.close()


def MSE_barplot():
    mse_values = []
    # xticks = []

    datasets = ["2d_cylinder_CFD", "2d_sq_cyl", "2d_plate", "channel_flow", "SST"]
    xticks = ["2d_cylinder", "2d_sq_cyl", "2d_plate", "channel_flow", "SST"]

    for i in datasets:
        try:
            mse = np.load(f"../results/{i}/MSE.npy")
            print(mse)
            mse_values.append(-1 * np.log(mse))
            # xticks.append(i)
        except:
            print(f"MSE for {i} not found")

    N = len(mse_values)  # number of the bars
    ind = np.arange(N)  # the x locations for the groups

    fig = plt.figure(figsize=[4, 3], dpi=600)
    width = 0.35
    ax = plt.gca()
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(width)
    tick_width = 0.35
    plt.tick_params(direction="in", width=tick_width)

    rects1 = ax.bar(
        ind, mse_values, width, color="blue", error_kw=dict(lw=1), capsize=2
    )
    w = 0.16
    ax.set_ylabel("Negative Log MSE per pixel", fontsize=8)
    plt.xticks(ind, xticks, rotation=0, fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylim((0, 12))

    # plt.xlabel('DL-ROM', fontsize=10)
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                1.003 * h,
                "%.2f" % float(h),
                ha="center",
                va="bottom",
                fontsize=5,
            )

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)
    plt.tick_params(direction="in", width=0.5)

    autolabel(rects1)

    plt.show()
    filename = "../results/MSE_barplot.png"
    fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d_set",
        dest="dset",
        type=str,
        default="2d_cylinder_CFD",
        help="Name of Dataset",
    )
    parser.add_argument(
        "-freq", dest="freq", type=int, default=20, help="Frequency for saving plots"
    )
    parser.add_argument("-mode", dest="mode", type=str, help="result/simulate")
    parser.add_argument("--MSE", dest="barplot", action="store_true")
    parser.add_argument("--train_plot", dest="train_plot", action="store_true")

    args = parser.parse_args()
    dataset_name = args.dset
    freq = args.freq
    mode = args.mode
    barplot = args.barplot
    train_plot = args.train_plot

    if not os.path.exists(f"../{mode}/{dataset_name}"):
        os.mkdir(f"../{mode}/{dataset_name}")

    if not os.path.exists(f"../{mode}/{dataset_name}/plots"):
        os.mkdir(f"../{mode}/{dataset_name}/plots")

    pred = np.load(f"../{mode}/{dataset_name}/predictions.npy")
    labels = np.load(f"../{mode}/{dataset_name}/labels.npy")

    # val_size, imageh, imagew
    if mode == "results":
        plot_results(pred, labels, mode, dataset_name, freq)

    elif mode == "simulate":
        mse = np.load(f"../{mode}/{dataset_name}/mse.npy")
        # plot_results(pred, labels, mode, dataset_name, freq)
        plot_simulate(mse)
        createAnimation(pred, dataset_name + "_pred")
        createAnimation(labels, dataset_name + "_ground_truth")
    else:
        print("No Mode Selected!")

    if barplot:
        MSE_barplot()

    if train_plot:
        plot_training_from_dict(dataset_name)
