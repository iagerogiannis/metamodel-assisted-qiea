import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from .list_operations import transpose_2d_list


def save_plot_data(filepath, data, plot_params, true_front=None):
    if not filepath:
        return

    base_name = os.path.splitext(filepath)[0]
    data_dir = base_name

    plot_type = plot_params["plot_type"]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset_labels = {}
    for i, d in enumerate(data, 1):
        dataset_name = f"dataset_{i}"
        csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
        dataset_labels[dataset_name] = d["label"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])

            if plot_type == "line":
                for x, y in zip(*d["data"]):
                    writer.writerow([x, y])
            elif plot_type == "scatter":
                for point in d["data"]:
                    writer.writerow(point)

    # Save dataset labels
    labels_path = os.path.join(data_dir, "dataset_labels.json")
    with open(labels_path, "w") as f:
        json.dump(dataset_labels, f, indent=2)

    # Save true front if provided
    if true_front is not None:
        true_front_path = os.path.join(data_dir, "true_front.csv")
        with open(true_front_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for point in true_front:
                writer.writerow(point)

    # Save plot parameters
    params_path = os.path.join(data_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(plot_params, f, indent=2)


def load_and_plot(folderpath):
    # Load parameters
    params_path = os.path.join(folderpath, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    # Load dataset labels
    labels_path = os.path.join(folderpath, "dataset_labels.json")
    with open(labels_path, "r") as f:
        dataset_labels = json.load(f)

    # Load datasets
    data = []
    for dataset_name in sorted(dataset_labels.keys()):
        csv_path = os.path.join(folderpath, f"{dataset_name}.csv")
        points = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                points.append([float(row[0]), float(row[1])])

        dataset = {"label": dataset_labels[dataset_name]}

        if params["plot_type"] == "line":
            points_array = np.array(points)
            dataset["data"] = [points_array[:, 0], points_array[:, 1]]
        else:
            dataset["data"] = points

        data.append(dataset)

    # Load true front if exists
    true_front = None
    true_front_path = os.path.join(folderpath, "true_front.csv")
    if os.path.exists(true_front_path):
        true_front = []
        with open(true_front_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                true_front.append([float(row[0]), float(row[1])])

    # Generate plot
    if params["plot_type"] == "line":
        line_chart(
            data,
            title=params.get("title", ""),
            xlabel=params.get("xlabel", ""),
            ylabel=params.get("ylabel", ""),
            logscale=params.get("logscale", False),
            withLegend=params.get("withLegend", False),
            xlim=params.get("xlim"),
            legendLoc=params.get("legendLoc", "best"),
        )
    elif params["plot_type"] == "scatter":
        scatter_plot(
            data,
            title=params.get("title", ""),
            xlabel=params.get("xlabel", ""),
            ylabel=params.get("ylabel", ""),
            withLegend=params.get("withLegend", False),
            connect_points=params.get("connect_points", False),
            true_front=true_front,
        )


def save_plot(filepath):
    if not filepath:
        return

    directory = os.path.dirname(filepath)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")


def plot_details(
    title,
    xlabel,
    ylabel,
    withLegend,
    logscale,
    xlim=None,
    legendLoc="best",
    title_fontsize=16,
    label_fontsize=15,
    legend_fontsize=13,
    legend_title_fontsize=14,
    legend_alignment=None,
    legend_markerfirst=True,
):
    if withLegend:
        legend_kwargs = {"loc": legendLoc}
        legend_kwargs["fontsize"] = legend_fontsize
        legend_kwargs["title_fontsize"] = legend_title_fontsize
        if legend_alignment is not None:
            legend_kwargs["alignment"] = legend_alignment
        legend_kwargs["markerfirst"] = legend_markerfirst
        plt.legend(**legend_kwargs)

    if title:
        plt.title(title, fontsize=title_fontsize)

    if xlabel:
        plt.xlabel(xlabel, fontsize=label_fontsize)

    if ylabel:
        plt.ylabel(ylabel, fontsize=label_fontsize)

    if logscale:
        plt.yscale("log")

    if xlim:
        padding_pct = 0.05
        padding = xlim * padding_pct
        plt.xlim(-padding, xlim + padding)


def line_chart(
    data,
    title="",
    xlabel="",
    ylabel="",
    logscale=False,
    withLegend=False,
    filepath="",
    xlim=None,
    legendLoc="best",
    save_data=False,
    title_fontsize=16,
    label_fontsize=15,
    legend_fontsize=13,
    legend_title_fontsize=14,
    legend_alignment=None,
    legend_markerfirst=True,
    params_text=None,
    params_fontsize=11,
    params_position="bottom_left",
):
    plt.clf()

    for d in data:
        linestyle = d.get("linestyle", "-")
        plt.plot(*d["data"], label=d["label"], linestyle=linestyle)

    plot_details(
        title,
        xlabel,
        ylabel,
        withLegend,
        logscale,
        xlim,
        legendLoc,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        legend_fontsize=legend_fontsize,
        legend_title_fontsize=legend_title_fontsize,
        legend_alignment=legend_alignment,
        legend_markerfirst=legend_markerfirst,
    )

    if params_text is not None:
        margin = 0.04

        if params_position == "top_right":
            x_pos = 1.0 - margin
            y_pos = 1.0 - margin
            ha = "right"
            va = "top"
        else:
            x_pos = margin
            y_pos = margin
            ha = "left"
            va = "bottom"

        text_obj = plt.gca().text(
            x_pos,
            y_pos,
            params_text,
            transform=plt.gca().transAxes,
            fontsize=params_fontsize,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6, pad=0.5),
            family="monospace",
            zorder=100,
            multialignment="left",
        )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(filepath)

    if save_data:
        save_plot_data(
            filepath,
            data,
            {
                "plot_type": "line",
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "logscale": logscale,
                "withLegend": withLegend,
                "xlim": xlim,
                "legendLoc": legendLoc,
            },
        )


def scatter_plot(
    data,
    title="",
    xlabel="",
    ylabel="",
    withLegend=False,
    filepath="",
    connect_points=False,
    true_front=None,
    save_data=False,
):
    plt.clf()

    for d in data:
        points_array = np.array(d["data"])

        if connect_points:
            sorted_indices = np.argsort(points_array[:, 0])
            points_array = points_array[sorted_indices]

        points = transpose_2d_list(points_array)

        if connect_points:
            plt.plot(*points, "o-", label=d["label"])
        else:
            plt.scatter(*points, label=d["label"], s=30)

    if true_front is not None:
        true_front_array = np.array(true_front)
        sorted_indices = np.argsort(true_front_array[:, 0])
        true_front_sorted = true_front_array[sorted_indices]
        points = transpose_2d_list(true_front_sorted)
        plt.plot(*points, "k-", linewidth=2, zorder=10)

    plot_details(title, xlabel, ylabel, withLegend, False)
    save_plot(filepath)
