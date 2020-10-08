# -*- coding: utf-8 -*-

"""Plotting module containing lots of plotting methods for the analysis notebook."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import chipsnet.utils


def save(name):
    """Save a matplotlib plot to file as both a pgf and pdf.

    Args:
        name (str): output path+name
    """
    plt.savefig("{}.pdf".format(name), bbox_inches="tight")
    plt.show()


def plot_cats(events, scale, cat_map, save_path):
    """Save a matplotlib plot to file as both a pgf and pdf.

    Args:
        events_u (pd.DataFrame): uniform events dataframe
        events_b (pd.DataFrame): beam events dataframe
        cat_map (dict): category map for this comparison
        save_path (str): path to save plot to
    """
    data = [
        len(events[events[cat_map["name"]] == i]) for i in range(len(cat_map["labels"]))
    ]
    data = [x * scale for x in data]
    cats = np.arange(len(cat_map["labels"]))
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.3})
    axs.bar(
        cats,
        data,
        color="tab:blue",
        width=1.0,
        label="uniform sample",
        edgecolor="black",
    )
    axs.set_xticks(cats)
    axs.set_xticklabels(cat_map["labels"], fontsize=24, rotation="vertical")
    axs.set_ylabel("Training events", fontsize=30)
    axs.set_ylim(10e2, 10e6)
    axs.set_yscale("log")
    save(save_path + "explore_" + cat_map["name"])


def plot_event(images_dict, event, save_name):
    """Plot hit, time and hough channels for an event.

    Args:
        images_dict (dict): images dictionary
        event (int): event to use
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 10), gridspec_kw={"wspace": 0.1})
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs[0].imshow(
        images_dict["r_charge_map_vtx"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
        vmin=0,
        vmax=255,
    )
    axs[0].set_title(r"Hit charge")
    axs[0].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[0].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[0].label_outer()

    axs[1].imshow(
        images_dict["r_time_map_vtx"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
        vmin=0,
        vmax=255,
    )
    axs[1].set_title(r"Hit time")
    axs[1].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[1].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[1].label_outer()

    neg = axs[2].imshow(
        images_dict["r_hough_map_vtx"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
        vmin=0,
        vmax=255,
    )
    axs[2].set_title(r"Hough value")
    axs[2].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[2].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[2].label_outer()
    fig.colorbar(neg, ax=axs, shrink=0.43, aspect=8, pad=0.025)
    save(save_name)


def plot_hit_time(images_dict, event, save_name):
    """Plot hit and time channels for the different image representations.

    Args:
        images_dict (dict): images dictionary
        event (int): event to use
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(
        2, 3, figsize=(16, 10), gridspec_kw={"hspace": 0.3, "wspace": 0.2}
    )
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs[0, 0].imshow(
        images_dict["r_charge_map_origin"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[0, 0].set_title(r"Origin view ($\phi$,$\theta$)")
    axs[0, 0].set_ylabel(r"$\theta$ bins", fontsize=30)

    axs[0, 1].imshow(
        images_dict["r_charge_map_iso"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[0, 1].set_title(r"Origin view ($x^{+}$,$x^{-}$)")
    axs[0, 1].set_ylabel(r"$x^{-}$ bins", fontsize=30)

    axs[0, 2].imshow(
        images_dict["r_charge_map_vtx"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[0, 2].set_title(r"Vertex view ($\phi$,$\theta$)")
    axs[0, 2].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[0, 2].text(68, 4, "Hit charge images", rotation=-90, fontsize=30)

    axs[1, 0].imshow(
        images_dict["r_time_map_origin"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[1, 0].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[1, 0].set_ylabel(r"$\theta$ bins", fontsize=30)

    axs[1, 1].imshow(
        images_dict["r_time_map_iso"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[1, 1].set_xlabel(r"$x^{+}$ bins", fontsize=30)
    axs[1, 1].set_ylabel(r"$x^{-}$ bins", fontsize=30)

    axs[1, 2].imshow(
        images_dict["r_time_map_vtx"][event],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[1, 2].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[1, 2].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[1, 2].text(68, 7, "Hit time images", rotation=-90, fontsize=30)
    save(save_name)


def plot_hough(images_dict, events, save_name):
    """Plot the hough channel representation.

    Args:
        images_dict (dict): images dictionary
        events (int): events to use
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"wspace": 0.2})
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs[0].imshow(
        images_dict["r_hough_map_vtx"][events[0]],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[0].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[0].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[0].label_outer()
    axs[1].imshow(
        images_dict["r_hough_map_vtx"][events[1]],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[1].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[1].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[1].label_outer()
    axs[2].imshow(
        images_dict["r_hough_map_vtx"][events[2]],
        cmap="Reds",
        origin="lower",
        extent=(0, 64, 0, 64),
    )
    axs[2].set_xlabel(r"$\phi$ bins", fontsize=30)
    axs[2].set_ylabel(r"$\theta$ bins", fontsize=30)
    axs[2].label_outer()
    save(save_name)


def plot_8bit_range(
    images_dict, max_charge=25, max_time=120, max_hough=3500, save_path=""
):
    """Plot the charge, time and hough channel 8-bit distributions.

    Args:
        images_dict (dict): images dictionary
        max_charge (float): maximum charge value
        max_time (float): maximum time value
        max_hough (float): maximum hough value
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    hist_data = []
    for event in images_dict["r_charge_map_vtx"]:
        hist_data.append(event.reshape(4096))
    hist_data = np.concatenate(hist_data, axis=0)
    occurrences_0 = np.count_nonzero(hist_data == 255) / len(hist_data)
    axs.hist(
        hist_data,
        range=(1, 256),
        bins=255,
        color="tab:green",
        histtype="step",
        linewidth=2,
        density=True,
    )
    hist_data = []
    for event in images_dict["r_time_map_vtx"]:
        hist_data.append(event.reshape(4096))
    hist_data = np.concatenate(hist_data, axis=0)
    occurrences_1 = np.count_nonzero(hist_data == 255) / len(hist_data)
    axs.hist(
        hist_data,
        range=(1, 256),
        bins=255,
        color="tab:blue",
        histtype="step",
        linewidth=2,
        density=True,
    )
    hist_data = []
    for event in images_dict["r_hough_map_vtx"]:
        hist_data.append(event.reshape(4096))
    hist_data = np.concatenate(hist_data, axis=0)
    occurrences_2 = np.count_nonzero(hist_data == 255) / len(hist_data)
    axs.hist(
        hist_data,
        range=(1, 256),
        bins=255,
        color="tab:red",
        histtype="step",
        linewidth=2,
        density=True,
    )
    print("[0,{}], outside range: {:.4f}".format(max_charge, occurrences_0))
    print("[0,{}], outside range: {:.4f}".format(max_time, occurrences_1))
    print("[0,{}], outside range: {:.4f}".format(max_hough, occurrences_2))

    axs.set_xlabel(r"8-bit value", fontsize=30)
    axs.set_ylabel(r"Frequency (arb.)", fontsize=30)
    axs.set_xlim(0, 260)
    axs.set_ylim(0, 0.06)

    hit = Line2D(
        [0],
        [0],
        color="tab:green",
        linewidth=2,
        linestyle="solid",
        label=r"Hit-charge",
    )
    time = Line2D(
        [0], [0], color="tab:blue", linewidth=2, linestyle="solid", label=r"Hit-time",
    )
    hough = Line2D(
        [0],
        [0],
        color="tab:red",
        linewidth=2,
        linestyle="solid",
        label=r"Hough-height",
    )
    axs.legend(handles=[hit, time, hough], loc="upper right", fontsize=28)
    save(save_path + "explore_8_bit_range")


def plot_cuts(config, events, save_path):
    """Plot the quality cuts.

    Args:
        config (dotmap.DotMap): configuration namespace
        events (pd.DataFrame): events dataframe
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(
        2, 2, figsize=(20, 15), gridspec_kw={"hspace": 0.2, "wspace": 0.1}
    )
    axs[0, 0].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 1)][
            "r_total_digi_q"
        ],
        color="tab:green",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[0, 0].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 0)][
            "r_total_digi_q"
        ],
        color="tab:olive",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[0, 0].hist(
        events[events.t_comb_cat == 1]["r_total_digi_q"],
        color="tab:blue",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{\mu}$ CC",
        linewidth=2,
    )
    axs[0, 0].hist(
        events[events.t_comb_cat == 2]["r_total_digi_q"],
        color="tab:red",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="NC",
        linewidth=2,
    )
    axs[0, 0].hist(
        events[events.t_comb_cat == 3]["r_total_digi_q"],
        color="black",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="cosmic",
        linewidth=2,
    )
    axs[0, 0].set_xlabel(r"Total charge (p.e)", fontsize=30)
    axs[0, 0].set_ylabel(r"Events (arb.)", fontsize=30)
    axs[0, 0].axvspan(0, config.eval.cuts.q, alpha=0.5, color="grey")
    axs[0, 0].set_yticks([])

    axs[0, 1].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 1)][
            "r_first_ring_height"
        ],
        color="tab:green",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[0, 1].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 0)][
            "r_first_ring_height"
        ],
        color="tab:olive",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[0, 1].hist(
        events[events.t_comb_cat == 1]["r_first_ring_height"],
        color="tab:blue",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{\mu}$ CC",
        linewidth=2,
    )
    axs[0, 1].hist(
        events[events.t_comb_cat == 2]["r_first_ring_height"],
        color="tab:red",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="NC",
        linewidth=2,
    )
    axs[0, 1].hist(
        events[events.t_comb_cat == 3]["r_first_ring_height"],
        color="black",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="cosmic",
        linewidth=2,
    )
    axs[0, 1].set_xlabel(r"Max Hough height (p.e)", fontsize=30)
    axs[0, 1].axvspan(0, config.eval.cuts.h, alpha=0.5, color="grey")
    axs[0, 1].set_yticks([])
    osc_nuel = Line2D(
        [0],
        [0],
        color="tab:green",
        linewidth=2,
        linestyle="solid",
        label=r"Appeared CC $\nu_{e}$",
    )
    nuel = Line2D(
        [0],
        [0],
        color="tab:olive",
        linewidth=2,
        linestyle="solid",
        label=r"Beam CC $\nu_{e}$",
    )
    numu = Line2D(
        [0],
        [0],
        color="tab:blue",
        linewidth=2,
        linestyle="solid",
        label=r"Survived CC $\nu_{\mu}$",
    )
    nc = Line2D([0], [0], color="tab:red", linewidth=2, linestyle="solid", label=r"NC")
    cosmic = Line2D(
        [0], [0], color="black", linewidth=2, linestyle="solid", label=r"Cosmic"
    )
    axs[0, 1].legend(
        handles=[osc_nuel, numu, nc, nuel, cosmic], loc="upper right", fontsize=28
    )

    axs[1, 0].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 1)]["r_dir_theta"],
        color="tab:green",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[1, 0].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 0)]["r_dir_theta"],
        color="tab:olive",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[1, 0].hist(
        events[events.t_comb_cat == 1]["r_dir_theta"],
        color="tab:blue",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="$\\nu_{\mu}$ CC",
        linewidth=2,
    )
    axs[1, 0].hist(
        events[events.t_comb_cat == 2]["r_dir_theta"],
        color="tab:red",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="NC",
        linewidth=2,
    )
    axs[1, 0].hist(
        events[events.t_comb_cat == 3]["r_dir_theta"],
        color="black",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="cosmic",
        linewidth=2,
    )
    axs[1, 0].set_xlabel(r"Seed $\theta$ direction (cos($\theta$))", fontsize=30)
    axs[1, 0].set_ylabel(r"Events (arb.)", fontsize=30)
    axs[1, 0].axvspan(-1, -config.eval.cuts.theta, alpha=0.5, color="grey")
    axs[1, 0].axvspan(config.eval.cuts.theta, 1, alpha=0.5, color="grey")
    axs[1, 0].set_ylim(0, 2.5)
    axs[1, 0].set_yticks([])

    axs[1, 1].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 1)]["r_dir_phi"]
        * 3.14159,
        color="tab:green",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[1, 1].hist(
        events[(events.t_comb_cat == 0) & (events.t_sample_type == 0)]["r_dir_phi"]
        * 3.14159,
        color="tab:olive",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="$\\nu_{e}$ CC",
        linewidth=2,
    )
    axs[1, 1].hist(
        events[events.t_comb_cat == 1]["r_dir_phi"] * 3.14159,
        color="tab:blue",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="$\\nu_{\mu}$ CC",
        linewidth=2,
    )
    axs[1, 1].hist(
        events[events.t_comb_cat == 2]["r_dir_phi"] * 3.14159,
        color="tab:red",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="NC",
        linewidth=2,
    )
    axs[1, 1].hist(
        events[events.t_comb_cat == 3]["r_dir_phi"] * 3.14159,
        color="black",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="cosmic",
        linewidth=2,
    )
    axs[1, 1].set_xlabel(r"Seed $\phi$ direction (radians)", fontsize=30)
    axs[1, 1].axvspan(-3.2, -config.eval.cuts.phi * 3.14159, alpha=0.5, color="grey")
    axs[1, 1].axvspan(config.eval.cuts.phi * 3.14159, 3.2, alpha=0.5, color="grey")
    axs[1, 1].set_yticks([])
    chipsnet.plotting.save(save_path + "explore_simple_cuts")


def plot_combined_values(events, type, prefix, save_path, cut=None):
    """Plot the output combined category values from the network.

    Args:
        events (pd.DataFrame): events dataframe
        prefix (str): prefix to choose correct model output
        save_path (str): path to save plot to
    """
    bins = 100
    hist_range = (0, 1)
    cat0 = prefix + "pred_t_comb_cat_0"
    cat1 = prefix + "pred_t_comb_cat_1"
    nuel_beam_cc_events = events[
        (events["t_comb_cat"] == 0)
        & (events["t_sample_type"] == 0)
        & (events["cut"] == 0)
    ]
    nuel_osc_cc_events = events[
        (events["t_comb_cat"] == 0)
        & (events["t_sample_type"] == 1)
        & (events["cut"] == 0)
    ]
    numu_cc_events = events[(events["t_comb_cat"] == 1) & (events["cut"] == 0)]
    nc_events = events[(events["t_comb_cat"] == 2) & (events["cut"] == 0)]
    cosmic_events = events[(events["t_comb_cat"] == 3) & (events["cut"] == 0)]

    if type == 0:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        plt.setp(axs, xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        axs.hist(
            nuel_beam_cc_events[cat0],
            weights=nuel_beam_cc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:olive",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            nuel_osc_cc_events[cat0],
            weights=nuel_osc_cc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:green",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            numu_cc_events[cat0],
            weights=numu_cc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:blue",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            nc_events[cat0],
            weights=nc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:red",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            cosmic_events[cat0],
            weights=cosmic_events["w"],
            range=hist_range,
            bins=bins,
            color="black",
            histtype="step",
            linewidth=2,
        )
        axs.set_ylim(0, 10)
        if cut is not None:
            axs.plot([cut, cut], [0, 4], "k-", lw=3, color="black")
            axs.arrow(
                cut + 0.005,
                4,
                0.05,
                0,
                color="black",
                lw=3,
                length_includes_head=True,
                head_width=0.5,
                head_length=0.02,
            )
        axs.set_xlabel(r"CC $\nu_{e}$ score", fontsize=30)
        axs.set_ylabel(r"Events/$6\times10^{20}$ POT", fontsize=30)
        nuel = Line2D(
            [0],
            [0],
            color="tab:olive",
            linewidth=2,
            linestyle="solid",
            label=r"Beam CC $\nu_{e}$",
        )
        osc_nuel = Line2D(
            [0],
            [0],
            color="tab:green",
            linewidth=2,
            linestyle="solid",
            label=r"Appeared CC $\nu_{e}$",
        )
        numu = Line2D(
            [0],
            [0],
            color="tab:blue",
            linewidth=2,
            linestyle="solid",
            label=r"Survived CC $\nu_{\mu}$",
        )
        nc = Line2D(
            [0], [0], color="tab:red", linewidth=2, linestyle="solid", label=r"NC"
        )
        # cosmic = Line2D(
        #    [0], [0], color="black", linewidth=2, linestyle="solid", label=r"cosmic"
        # )
        axs.legend(handles=[osc_nuel, nuel, nc, numu], loc="upper center", fontsize=28)

    if type == 1:
        fig, axs = plt.subplots(
            1, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.1, "wspace": 0.1}
        )
        axs.hist(
            nuel_beam_cc_events[cat1],
            weights=nuel_beam_cc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:olive",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            nuel_osc_cc_events[cat1],
            weights=nuel_osc_cc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:green",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            numu_cc_events[cat1],
            weights=numu_cc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:blue",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            nc_events[cat1],
            weights=nc_events["w"],
            range=hist_range,
            bins=bins,
            color="tab:red",
            histtype="step",
            linewidth=2,
        )
        axs.hist(
            cosmic_events[cat1],
            weights=cosmic_events["w"],
            range=hist_range,
            bins=bins,
            color="black",
            histtype="step",
            linewidth=2,
        )
        axs.set_ylim(10e-3, 10e2)
        if cut is not None:
            axs.plot([cut, cut], [10e-3, 5e0], "k-", lw=3, color="black")
            axs.arrow(
                cut,
                5e0,
                0.05,
                0,
                color="black",
                lw=3,
                length_includes_head=True,
                head_width=3,
                head_length=0.02,
            )
        # axs.set_ylim(0, 500)
        axs.set_xlabel(r"CC $\nu_{\mu}$ score", fontsize=30)
        axs.set_ylabel(r"Events/$6\times10^{20}$ POT", fontsize=30)
        axs.set_yscale("log")
        nuel = Line2D(
            [0],
            [0],
            color="tab:olive",
            linewidth=2,
            linestyle="solid",
            label=r"Beam CC $\nu_{e}$",
        )
        osc_nuel = Line2D(
            [0],
            [0],
            color="tab:green",
            linewidth=2,
            linestyle="solid",
            label=r"Appeared CC $\nu_{e}$",
        )
        numu = Line2D(
            [0],
            [0],
            color="tab:blue",
            linewidth=2,
            linestyle="solid",
            label=r"Survived CC $\nu_{\mu}$",
        )
        nc = Line2D(
            [0], [0], color="tab:red", linewidth=2, linestyle="solid", label=r"NC"
        )
        # cosmic = Line2D(
        #    [0], [0], color="black", linewidth=2, linestyle="solid", label=r"cosmic"
        # )
        axs.legend(handles=[numu, nc, osc_nuel, nuel], loc="upper center", fontsize=28)

    save(save_path)


def plot_cosmic_values(events, prefix, save_path, zoom=False):
    """Plot the output cosmic rejection values from the network.

    Args:
        events (pd.DataFrame): events dataframe
        prefix (str): prefix to choose correct model output
        save_path (str): path to save plot to
    """
    hist_range = (0, 1)
    key = prefix + "_pred_t_cosmic_cat"
    nuel_beam_cc_events = events[
        (events["t_comb_cat"] == 0)
        & (events["t_sample_type"] == 0)
        & (events["simple_cut"] == 0)
    ]
    nuel_osc_cc_events = events[
        (events["t_comb_cat"] == 0)
        & (events["t_sample_type"] == 1)
        & (events["simple_cut"] == 0)
    ]
    numu_cc_events = events[(events["t_comb_cat"] == 1) & (events["simple_cut"] == 0)]
    nc_events = events[(events["t_comb_cat"] == 2) & (events["simple_cut"] == 0)]
    cosmic_events = events[(events["t_comb_cat"] == 3) & (events["simple_cut"] == 0)]
    fig, axs = plt.subplots(
        1, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.1, "wspace": 0.1}
    )

    bins = 25
    hist_range = (0, 1)
    plt.setp(axs, xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    axs.set_xlim(hist_range[0], hist_range[1])
    axs.set_ylim(10e-4, 10e5)
    if zoom:
        bins = 50
        hist_range = (0, 0.001)
        plt.setp(axs, xticks=[0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001])
        axs.set_xlim(hist_range[0], hist_range[1])
        axs.set_ylim(10e-4, 10e3)
        axs.plot([0.0001, 0.0001], [10e-4, 10e1], "k-", lw=3, color="black")
        axs.arrow(
            0.0001,
            10e1,
            -0.00005,
            0,
            color="black",
            lw=3,
            length_includes_head=True,
            head_width=35,
            head_length=0.00002,
        )
    axs.hist(
        nuel_beam_cc_events[key],
        weights=nuel_beam_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:olive",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        nuel_osc_cc_events[key],
        weights=nuel_osc_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:green",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        numu_cc_events[key],
        weights=numu_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:blue",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        nc_events[key],
        weights=nc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:red",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        cosmic_events[key],
        weights=cosmic_events["w"],
        range=hist_range,
        bins=bins,
        color="black",
        histtype="step",
        linewidth=2,
    )
    axs.set_yscale("log")
    axs.set_xlabel(r"Cosmic score", fontsize=30)
    axs.set_ylabel(r"Events/$6\times10^{20}$ POT", fontsize=30)
    nuel = Line2D(
        [0],
        [0],
        color="tab:olive",
        linewidth=2,
        linestyle="solid",
        label=r"Beam CC $\nu_{e}$",
    )
    osc_nuel = Line2D(
        [0],
        [0],
        color="tab:green",
        linewidth=2,
        linestyle="solid",
        label=r"Appeared CC $\nu_{e}$",
    )
    numu = Line2D(
        [0],
        [0],
        color="tab:blue",
        linewidth=2,
        linestyle="solid",
        label=r"Survived CC $\nu_{\mu}$",
    )
    nc = Line2D([0], [0], color="tab:red", linewidth=2, linestyle="solid", label=r"NC")
    cosmic = Line2D(
        [0], [0], color="black", linewidth=2, linestyle="solid", label=r"Cosmic"
    )
    axs.legend(
        handles=[cosmic, osc_nuel, numu, nuel, nc], loc="upper center", fontsize=26
    )

    save(save_path)


def plot_escapes_values(events, prefix, save_path):
    """Plot the output escapes values from the network.

    Args:
        events (pd.DataFrame): events dataframe
        prefix (str): prefix to choose correct model output
        save_path (str): path to save plot to
    """
    bins = 50
    hist_range = (0, 1)
    key = prefix + "_pred_t_escapes"
    nuel_beam_cc_events = events[
        (events["t_comb_cat"] == 0)
        & (events["t_sample_type"] == 0)
        & (events["simple_cut"] == 0)
        & (events["cosmic_cut"] == 0)
    ]
    nuel_osc_cc_events = events[
        (events["t_comb_cat"] == 0)
        & (events["t_sample_type"] == 1)
        & (events["simple_cut"] == 0)
        & (events["cosmic_cut"] == 0)
    ]
    numu_cc_events = events[
        (events["t_comb_cat"] == 1)
        & (events["simple_cut"] == 0)
        & (events["cosmic_cut"] == 0)
    ]
    nc_events = events[
        (events["t_comb_cat"] == 2)
        & (events["simple_cut"] == 0)
        & (events["cosmic_cut"] == 0)
    ]
    cosmic_events = events[
        (events["t_comb_cat"] == 3)
        & (events["simple_cut"] == 0)
        & (events["cosmic_cut"] == 0)
    ]

    fig, axs = plt.subplots(
        1, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.1, "wspace": 0.1}
    )
    plt.setp(axs, xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    axs.hist(
        nuel_beam_cc_events[key],
        weights=nuel_beam_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:olive",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        nuel_osc_cc_events[key],
        weights=nuel_osc_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:green",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        numu_cc_events[key],
        weights=numu_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:blue",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        nc_events[key],
        weights=nc_events["w"],
        range=hist_range,
        bins=bins,
        color="tab:red",
        histtype="step",
        linewidth=2,
    )
    axs.hist(
        cosmic_events[key],
        weights=cosmic_events["w"],
        range=hist_range,
        bins=bins,
        color="black",
        histtype="step",
        linewidth=2,
    )
    axs.plot([0.5, 0.5], [10e-3, 5e0], "k-", lw=3, color="black")
    axs.arrow(
        0.5,
        5e0,
        -0.05,
        0,
        color="black",
        lw=3,
        length_includes_head=True,
        head_width=3,
        head_length=0.02,
    )
    axs.set_ylim(10e-3, 10e2)
    axs.set_yscale("log")
    axs.set_xlabel(r"Escapes score", fontsize=30)
    axs.set_ylabel(r"Events/$6\times10^{20}$ POT", fontsize=30)
    nuel = Line2D(
        [0],
        [0],
        color="tab:olive",
        linewidth=2,
        linestyle="solid",
        label=r"Beam CC $\nu_{e}$",
    )
    osc_nuel = Line2D(
        [0],
        [0],
        color="tab:green",
        linewidth=2,
        linestyle="solid",
        label=r"Appeared CC $\nu_{e}$",
    )
    numu = Line2D(
        [0],
        [0],
        color="tab:blue",
        linewidth=2,
        linestyle="solid",
        label=r"Survived CC $\nu_{\mu}$",
    )
    nc = Line2D([0], [0], color="tab:red", linewidth=2, linestyle="solid", label=r"NC")
    # cosmic = Line2D(
    #    [0], [0], color="black", linewidth=2, linestyle="solid", label=r"cosmic"
    # )
    axs.legend(handles=[osc_nuel, numu, nuel, nc], loc="upper center", fontsize=26)

    save(save_path)


def plot_eff_curves(events, cat, save_path, full=False, leg_pos=None):
    """Plot the eff, pur, fom curves.

    Args:
        events (dict): events output
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    plt.setp(
        axs, xticks=[0, 0.20, 0.40, 0.60, 0.80, 1], yticks=[0, 20, 40, 60, 80, 100]
    )
    if cat == 1 and full is False:
        plt.setp(
            axs,
            xticks=[0, 0.20, 0.40, 0.60, 0.80, 1],
            yticks=[0, 75, 80, 85, 90, 95, 100],
        )
    styles = ["solid", "dashed", "dotted", "dashdot"]
    for i in range(len(events)):
        axs.plot(
            events[i]["cuts"],
            events[i]["sig_effs"][cat] * 100,
            color="tab:orange",
            linestyle=styles[i],
            label="",
            linewidth=2,
        )
        axs.plot(
            events[i]["cuts"],
            events[i]["purs"][cat] * 100,
            color="tab:purple",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.plot(
            events[i]["cuts"],
            events[i]["foms_0"][cat] * 100,
            color="tab:cyan",
            linestyle=styles[i],
            linewidth=3,
        )
    axs.set_xlabel(r"CC $\nu_{e}$ score", fontsize=30)
    if cat == 1:
        axs.set_xlabel(r"CC $\nu_{\mu}$ score", fontsize=30)
    axs.set_ylabel("Percentage", fontsize=30)
    axs.set_ylim(0, 100)
    if cat == 1 and full is False:
        axs.set_ylim(70, 100)
    axs.set_xlim(0, 1)
    signal = Line2D(
        [0],
        [0],
        color="tab:orange",
        linewidth=2,
        linestyle="solid",
        label=r"CC $\nu_{e}$ efficiency",
    )
    if cat == 1:
        signal = Line2D(
            [0],
            [0],
            color="tab:orange",
            linewidth=2,
            linestyle="solid",
            label=r"CC $\nu_{\mu}$ efficiency",
        )
    pur = Line2D(
        [0],
        [0],
        color="tab:purple",
        linewidth=2,
        linestyle="solid",
        label=r"CC $\nu_{e}$ purity",
    )
    if cat == 1:
        pur = Line2D(
            [0],
            [0],
            color="tab:purple",
            linewidth=2,
            linestyle="solid",
            label=r"CC $\nu_{\mu}$ purity",
        )
    fom = Line2D(
        [0],
        [0],
        color="tab:cyan",
        linewidth=3,
        linestyle="solid",
        label=r"Efficiency $\times$ purity",
    )
    if cat == 0:
        if leg_pos is not None:
            axs.legend(handles=[signal, pur, fom], loc=leg_pos, fontsize=28)
        else:
            axs.legend(handles=[signal, pur, fom], loc="center left", fontsize=28)
    elif cat == 1:
        if leg_pos is not None:
            axs.legend(handles=[signal, pur, fom], loc=leg_pos, fontsize=28)
        else:
            axs.legend(handles=[signal, pur, fom], loc="upper right", fontsize=28)
    axs.grid()
    save(save_path)


def plot_comp_curves(events, cat, save_path):
    """Plot the eff, pur, fom curves.

    Args:
        events (dict): events output
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.3})
    # plt.setp(axs, xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.25, 0.5, 0.75, 1])
    styles = ["solid", "dashed", "dotted", "dashdot"]

    for i in range(len(events)):
        axs[0].plot(
            events[i]["bkg_effs"][cat],
            events[i]["sig_effs"][cat],
            color="tab:pink",
            linestyle=styles[i],
            linewidth=2,
        )
    axs[0].set_xlabel("Background efficiency", fontsize=30)
    if cat == 0:
        axs[0].set_ylabel(r"CC $\nu_{e}$ efficiency", fontsize=30)
    else:
        axs[0].set_ylabel(r"CC $\nu_{\mu}$ efficiency", fontsize=30)
    # axs[0].set_ylim(0.6, 1)
    # axs[0].set_xlim(0, 0.25)
    axs[0].legend()
    axs[0].grid()
    for i in range(len(events)):
        axs[1].plot(
            events[i]["sig_effs"][cat],
            events[i]["purs"][cat],
            color="tab:brown",
            linestyle=styles[i],
            linewidth=2,
        )
    if cat == 0:
        axs[1].set_xlabel(r"CC $\nu_{e}$ efficiency", fontsize=30)
        axs[1].set_ylabel(r"CC $\nu_{e}$ Purity", fontsize=30)
    else:
        axs[1].set_xlabel(r"CC $\nu_{\mu}$ efficiency", fontsize=30)
        axs[1].set_ylabel(r"CC $\nu_{\mu}$ Purity", fontsize=30)
    # axs[1].set_ylim(0, 1)
    # axs[1].set_xlim(0, 1)
    axs[1].legend()
    axs[1].grid()
    save(save_path)


def plot_nuel_hists(events, ev, save_path, energy=r"E (GeV)"):
    """Plot the eff and pur plot vs neutrino energy.

    Args:
        events (dict): events output
        ev (pd.DataFrame): events DataFrame
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    bins = np.arange(0.25, 10.25, 0.5)
    styles = ["solid", "dashed", "dotted", "dashdot"]

    for i in range(len(events)):
        axs.errorbar(
            bins,
            events[i]["fom_effs"][0][0][0] * 100,
            yerr=events[i]["fom_effs"][0][0][1] * 100,
            color="tab:green",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_effs"][0][1][0] * 100,
            yerr=events[i]["fom_effs"][0][1][1] * 100,
            color="tab:olive",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_effs"][0][2][0] * 100,
            yerr=events[i]["fom_effs"][0][2][1] * 100,
            color="tab:blue",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_effs"][0][3][0] * 100,
            yerr=events[i]["fom_effs"][0][3][1] * 100,
            color="tab:red",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_purs"][0][0] * 100,
            yerr=events[i]["fom_purs"][0][1] * 100,
            color="black",
            linestyle=styles[i],
            linewidth=2,
        )
    axs.hist(
        ev[(ev["t_comb_cat"] == 0) & (ev["t_sample_type"] == 1)]["t_nu_energy"] / 1000,
        range=(1, 8),
        bins=14,
        color="tab:green",
        density=False,
        alpha=0.3,
        weights=ev[(ev["t_comb_cat"] == 0) & (ev["t_sample_type"] == 1)]["w"] * 6,
    )
    axs.set_xlabel(energy, fontsize=30)
    axs.set_ylabel("Percentage", fontsize=30)
    axs.set_ylim([0, 100])
    axs.set_xlim([0.5, 10])
    axs.grid()
    nuel = Line2D(
        [0],
        [0],
        color="tab:olive",
        linewidth=2,
        linestyle="solid",
        label=r"Beam CC $\nu_{e}$ efficiency",
    )
    osc_nuel = Line2D(
        [0],
        [0],
        color="tab:green",
        linewidth=2,
        linestyle="solid",
        label=r"Appeared CC $\nu_{e}$ efficiency",
    )
    numu = Line2D(
        [0],
        [0],
        color="tab:blue",
        linewidth=2,
        linestyle="solid",
        label=r"Survived CC $\nu_{\mu}$ efficiency",
    )
    nc = Line2D(
        [0],
        [0],
        color="tab:red",
        linewidth=2,
        linestyle="solid",
        label=r"NC efficiency",
    )
    purity = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2,
        linestyle="solid",
        label=r"Appeared CC $\nu_{e}$ purity",
    )
    axs.legend(
        handles=[osc_nuel, numu, nuel, nc, purity], loc="center right", fontsize=24
    )
    save(save_path)


def plot_numu_hists(events, ev, save_path, energy=r"E (GeV)"):
    """Plot the eff and pur plot vs neutrino energy.

    Args:
        events (dict): events output
        ev (pd.DataFrame): events DataFrame
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    bins = np.arange(0.25, 10.25, 0.5)
    styles = ["solid", "dashed", "dotted", "dashdot"]

    for i in range(len(events)):
        axs.errorbar(
            bins,
            events[i]["fom_effs"][2][0][0] * 100,
            yerr=events[i]["fom_effs"][2][0][1] * 100,
            color="tab:green",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_effs"][2][1][0] * 100,
            yerr=events[i]["fom_effs"][2][1][1] * 100,
            color="tab:olive",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_effs"][2][2][0] * 100,
            yerr=events[i]["fom_effs"][2][2][1] * 100,
            color="tab:blue",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_effs"][2][3][0] * 100,
            yerr=events[i]["fom_effs"][2][3][1] * 100,
            color="tab:red",
            linestyle=styles[i],
            linewidth=2,
        )
        axs.errorbar(
            bins,
            events[i]["fom_purs"][2][0] * 100,
            yerr=events[i]["fom_purs"][2][1] * 100,
            color="black",
            linestyle=styles[i],
            linewidth=2,
        )
    axs.hist(
        ev[ev["t_comb_cat"] == 1]["t_nu_energy"] / 1000,
        range=(1, 8),
        bins=14,
        color="tab:blue",
        density=False,
        alpha=0.3,
        weights=ev[ev["t_comb_cat"] == 1]["w"] * 0.15,
    )
    axs.set_xlabel(energy, fontsize=30)
    axs.set_ylabel("Percentage", fontsize=30)
    axs.set_ylim([0, 100])
    axs.set_xlim([0.5, 10])
    axs.grid()
    nuel = Line2D(
        [0],
        [0],
        color="tab:olive",
        linewidth=2,
        linestyle="solid",
        label=r"Beam CC $\nu_{e}$ efficiency",
    )
    osc_nuel = Line2D(
        [0],
        [0],
        color="tab:green",
        linewidth=2,
        linestyle="solid",
        label=r"Appeared CC $\nu_{e}$ efficiency",
    )
    numu = Line2D(
        [0],
        [0],
        color="tab:blue",
        linewidth=2,
        linestyle="solid",
        label=r"Survived CC $\nu_{\mu}$ efficiency",
    )
    nc = Line2D(
        [0],
        [0],
        color="tab:red",
        linewidth=2,
        linestyle="solid",
        label=r"NC efficiency",
    )
    purity = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2,
        linestyle="solid",
        label=r"Survived CC $\nu_{\mu}$ purity",
    )
    axs.legend(
        handles=[osc_nuel, numu, nuel, nc, purity], loc="center right", fontsize=24
    )
    save(save_path)


def plot_history(
    config, model, save_path, key="accuracy", label=r"accuracy", type="max"
):
    """Plot the training history of a list of models.

    Args:
        config (dotmap.DotMap): configuration namespace
        models (list[str]): list of models to plot
        save_path (str): path to save plot to
        key (str): Key to use for second metric alongside loss
        label (str): Label to use
        type (str): max or min, which is best?
    """
    fig, axs = plt.subplots(
        1, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.2, "wspace": 0.3}
    )
    # plt.setp(axs, xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    history = chipsnet.utils.model_history(config, model)
    epochs = np.arange(1, len(history["loss"]) + 1)
    axs.set_xlabel("Epoch", fontsize=32)
    axs.set_ylabel(label, color="tab:red", fontsize=32)
    axs.plot(epochs, history[key], color="tab:red", linestyle="solid", linewidth=2)
    axs.plot(
        epochs, history["val_" + key], color="tab:red", linestyle="dashed", linewidth=2
    )
    best_epoch = history["val_" + key].to_numpy().argmax() + 1
    if type == "min":
        best_epoch = history["val_" + key].to_numpy().argmin() + 1
    axs.plot([best_epoch, best_epoch], axs.get_ylim(), "k-", lw=4)

    axs.tick_params(axis="y", labelcolor="tab:red")
    axs_t = axs.twinx()  # instantiate a second axes that shares the same x-axis
    axs_t.set_ylabel(
        "Loss", color="tab:blue", fontsize=32
    )  # we already handled the x-label with ax1
    axs_t.plot(
        epochs, history["loss"], color="tab:blue", linestyle="solid", linewidth=2
    )
    axs_t.plot(
        epochs, history["val_loss"], color="tab:blue", linestyle="dashed", linewidth=2
    )
    axs_t.tick_params(axis="y", labelcolor="tab:blue")
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    save(save_path)
