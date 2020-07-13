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
    plt.savefig("{}.pgf".format(name))
    plt.savefig("{}.pdf".format(name))
    plt.show()


def plot_cats(events_u, events_b, cat_map, save_path):
    """Save a matplotlib plot to file as both a pgf and pdf.

    Args:
        events_u (pd.DataFrame): uniform events dataframe
        events_b (pd.DataFrame): beam events dataframe
        cat_map (dict): category map for this comparison
        save_path (str): path to save plot to
    """
    data_u = [
        len(events_u[events_u[cat_map["name"]] == i])
        for i in range(len(cat_map["labels"]))
    ]
    data_b = [
        len(events_b[events_b[cat_map["name"]] == i])
        for i in range(len(cat_map["labels"]))
    ]
    cats = np.arange(len(cat_map["labels"]))
    width = 0.5

    fig, axs = plt.subplots(1, 1, figsize=(18, 5), gridspec_kw={"hspace": 0.3})
    axs.bar(
        cats + width / 2,
        data_u,
        color="royalblue",
        width=width,
        label="uniform sample",
        edgecolor="black",
    )
    axs.bar(
        cats - width / 2,
        data_b,
        color="firebrick",
        width=width,
        label="flux sample",
        edgecolor="black",
    )
    axs.set_xticks(cats)
    axs.set_xticklabels(cat_map["labels"], fontsize=14, rotation="vertical")
    axs.set_ylabel("Testing events")
    axs.legend()
    save(save_path + cat_map["name"])


def plot_hit_time(images_dict, event, save_name):
    """Plot hit and time channels for the different image representations.

    Args:
        images_dict (dict): images dictionary
        event (int): event to use
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(
        2, 3, figsize=(15, 10), gridspec_kw={"hspace": 0.3, "wspace": 0.2}
    )
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs[0, 0].imshow(
        images_dict["r_charge_map_origin"][event], cmap="Reds", origin="lower"
    )
    axs[0, 0].set_title(r"origin view ($\phi$,$\theta$)")
    axs[0, 0].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")

    axs[0, 1].imshow(
        images_dict["r_charge_map_iso"][event], cmap="Reds", origin="lower"
    )
    axs[0, 1].set_title(r"origin view ($x^{+}$,$x^{-}$)")
    axs[0, 1].set(xlabel=r"$x^{+}$ bins", ylabel=r"$x^{-}$ bins")

    axs[0, 2].imshow(
        images_dict["r_charge_map_vtx"][event], cmap="Reds", origin="lower"
    )
    axs[0, 2].set_title(r"vertex view ($\phi$,$\theta$)")
    axs[0, 2].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs[0, 2].text(68, 2, "Hit charge images", rotation=-90, fontsize=24)

    axs[1, 0].imshow(
        images_dict["r_time_map_origin"][event], cmap="Reds", origin="lower"
    )
    axs[1, 0].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")

    axs[1, 1].imshow(images_dict["r_time_map_iso"][event], cmap="Reds", origin="lower")
    axs[1, 1].set(xlabel=r"$x^{+}$ bins", ylabel=r"$x^{-}$ bins")

    axs[1, 2].imshow(images_dict["r_time_map_vtx"][event], cmap="Reds", origin="lower")
    axs[1, 2].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs[1, 2].text(68, 8, "First hit time images", rotation=-90, fontsize=24)
    save(save_name)


def plot_hough(images_dict, events, save_name):
    """Plot the hough channel representation.

    Args:
        images_dict (dict): images dictionary
        events (int): events to use
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(
        1, 3, figsize=(15, 5), gridspec_kw={"hspace": 0.3, "wspace": 0.2}
    )
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs[0].imshow(
        images_dict["r_hough_map_vtx"][events[0]], cmap="Reds", origin="lower"
    )
    axs[0].set_title(r"vertex view ($\phi$,$\theta$)")
    axs[0].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs[0].label_outer()
    axs[1].imshow(
        images_dict["r_hough_map_vtx"][events[1]], cmap="Reds", origin="lower"
    )
    axs[1].set_title(r"vertex view ($\phi$,$\theta$)")
    axs[1].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs[1].label_outer()
    axs[2].imshow(
        images_dict["r_hough_map_vtx"][events[2]], cmap="Reds", origin="lower"
    )
    axs[2].set_title(r"vertex view ($\phi$,$\theta$)")
    axs[2].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs[2].label_outer()
    axs[2].text(68, 10, "Hough space image", rotation=-90, fontsize=24)
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
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    hist_data = []
    for event in images_dict["r_charge_map_vtx"]:
        hist_data.append(event.reshape(4096))
    hist_data = np.concatenate(hist_data, axis=0)
    occurrences = np.count_nonzero(hist_data == 255) / len(hist_data)
    axs[0].hist(hist_data, range=(1, 256), bins=255, color="black", histtype="step")
    axs[0].set_title(
        "[0,{}], outside range: {:.4f}".format(max_charge, occurrences), fontsize=17
    )
    axs[0].set(xlabel="Hit charge 8-bit value", ylabel="Frequency")
    axs[0].label_outer()

    hist_data = []
    for event in images_dict["r_time_map_vtx"]:
        hist_data.append(event.reshape(4096))
    hist_data = np.concatenate(hist_data, axis=0)
    occurrences = np.count_nonzero(hist_data == 255) / len(hist_data)
    axs[1].hist(hist_data, range=(1, 256), bins=255, color="black", histtype="step")
    axs[1].set_title(
        "[0,{}], outside range: {:.4f}".format(max_time, occurrences), fontsize=17
    )
    axs[1].set(xlabel="Hit time 8-bit value", ylabel="Frequency")
    axs[1].label_outer()

    hist_data = []
    for event in images_dict["r_hough_map_vtx"]:
        hist_data.append(event.reshape(4096))
    hist_data = np.concatenate(hist_data, axis=0)
    occurrences = np.count_nonzero(hist_data == 255) / len(hist_data)
    axs[2].hist(hist_data, range=(1, 256), bins=255, color="black", histtype="step")
    axs[2].set_title(
        "[0,{}], outside range: {:.4f}".format(max_hough, occurrences), fontsize=17
    )
    axs[2].set(xlabel="Hough 8-bit value", ylabel="Frequency")
    axs[2].label_outer()

    save(save_path + "8_bit_range")


def plot_cuts(config, events, save_path):
    """Plot the quality cuts.

    Args:
        config (dotmap.DotMap): configuration namespace
        events (pd.DataFrame): events dataframe
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(
        2, 2, figsize=(12, 10), gridspec_kw={"hspace": 0.4, "wspace": 0.1}
    )
    axs[0, 0].hist(
        events[events.t_comb_cat == 0]["r_total_digi_q"],
        color="green",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{e}$ CC",
    )
    axs[0, 0].hist(
        events[events.t_comb_cat == 1]["r_total_digi_q"],
        color="blue",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{\mu}$ CC",
    )
    axs[0, 0].hist(
        events[events.t_comb_cat == 2]["r_total_digi_q"],
        color="red",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="NC",
    )
    axs[0, 0].hist(
        events[events.t_comb_cat == 3]["r_total_digi_q"],
        color="black",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="Cosmic",
    )
    axs[0, 0].set(xlabel="Total collected charge (p.e)")
    axs[0, 0].axvspan(0, config.eval.cuts.q, alpha=0.5, color="grey")
    axs[0, 0].set_yticks([])
    # axs[0, 0].legend()

    axs[0, 1].hist(
        events[events.t_comb_cat == 0]["r_first_ring_height"],
        color="green",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{e}$ CC",
    )
    axs[0, 1].hist(
        events[events.t_comb_cat == 1]["r_first_ring_height"],
        color="blue",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="$\\nu_{\mu}$ CC",
    )
    axs[0, 1].hist(
        events[events.t_comb_cat == 2]["r_first_ring_height"],
        color="red",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="NC",
    )
    axs[0, 1].hist(
        events[events.t_comb_cat == 3]["r_first_ring_height"],
        color="black",
        histtype="step",
        range=(0, 5000),
        bins=40,
        density=True,
        label="Cosmic",
    )
    axs[0, 1].set(xlabel="Hough ring height (p.e)")
    axs[0, 1].axvspan(0, config.eval.cuts.h, alpha=0.5, color="grey")
    axs[0, 1].set_yticks([])
    # axs[0, 1].legend()

    axs[1, 0].hist(
        events[events.t_comb_cat == 0]["r_dir_theta"],
        color="green",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="$\\nu_{e}$ CC",
    )
    axs[1, 0].hist(
        events[events.t_comb_cat == 1]["r_dir_theta"],
        color="blue",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="$\\nu_{\mu}$ CC",
    )
    axs[1, 0].hist(
        events[events.t_comb_cat == 2]["r_dir_theta"],
        color="red",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="NC",
    )
    axs[1, 0].hist(
        events[events.t_comb_cat == 3]["r_dir_theta"],
        color="black",
        histtype="step",
        range=(-1, 1),
        bins=64,
        density=True,
        label="Cosmic",
    )
    axs[1, 0].set(xlabel=r"Reco $\theta$ direction (cos($\theta$))")
    axs[1, 0].axvspan(-1, -config.eval.cuts.theta, alpha=0.5, color="grey")
    axs[1, 0].axvspan(config.eval.cuts.theta, 1, alpha=0.5, color="grey")
    axs[1, 0].set_yticks([])
    # axs[1, 0].legend()

    axs[1, 1].hist(
        events[events.t_comb_cat == 0]["r_dir_phi"] * 3.14159,
        color="green",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="$\\nu_{e}$ CC",
    )
    axs[1, 1].hist(
        events[events.t_comb_cat == 1]["r_dir_phi"] * 3.14159,
        color="blue",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="$\\nu_{\mu}$ CC",
    )
    axs[1, 1].hist(
        events[events.t_comb_cat == 2]["r_dir_phi"] * 3.14159,
        color="red",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="NC",
    )
    axs[1, 1].hist(
        events[events.t_comb_cat == 3]["r_dir_phi"] * 3.14159,
        color="black",
        histtype="step",
        range=(-3.2, 3.2),
        bins=64,
        density=True,
        label="Cosmic",
    )
    axs[1, 1].set(xlabel=r"Reco $\phi$ direction (rads)")
    axs[1, 1].axvspan(-3.2, -config.eval.cuts.phi * 3.14159, alpha=0.5, color="grey")
    axs[1, 1].axvspan(config.eval.cuts.phi * 3.14159, 3.2, alpha=0.5, color="grey")
    axs[1, 1].set_yticks([])
    # axs[1, 1].legend()
    chipsnet.plotting.save(save_path + "simple_cuts")


def plot_combined_values(events, prefix, save_path):
    """Plot the output combined category values from the network.

    Args:
        events (pd.DataFrame): events dataframe
        prefix (str): prefix to choose correct model output
        save_path (str): path to save plot to
    """
    bins = 25
    hist_range = (0, 1)
    cat0 = prefix + "pred_t_comb_cat_0"
    cat1 = prefix + "pred_t_comb_cat_1"
    cat2 = prefix + "pred_t_comb_cat_2"
    nuel_cc_events = events[events["t_comb_cat"] == 0]
    numu_cc_events = events[events["t_comb_cat"] == 1]
    nc_events = events[events["t_comb_cat"] == 2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(
        nuel_cc_events[cat0],
        weights=nuel_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="green",
        histtype="step",
    )
    axs[0].hist(
        numu_cc_events[cat0],
        weights=numu_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="blue",
        histtype="step",
    )
    axs[0].hist(
        nc_events[cat0],
        weights=nc_events["w"],
        range=hist_range,
        bins=bins,
        color="red",
        histtype="step",
    )
    axs[0].set_xlabel("Combined nuel cc score", fontsize=17)
    axs[0].set_yscale("log")

    axs[1].hist(
        nuel_cc_events[cat1],
        weights=nuel_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="green",
        histtype="step",
    )
    axs[1].hist(
        numu_cc_events[cat1],
        weights=numu_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="blue",
        histtype="step",
    )
    axs[1].hist(
        nc_events[cat1],
        weights=nc_events["w"],
        range=hist_range,
        bins=bins,
        color="red",
        histtype="step",
    )
    axs[1].set_xlabel("Combined numu cc score", fontsize=17)
    axs[1].set_yscale("log")

    axs[2].hist(
        nuel_cc_events[cat2],
        weights=nuel_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="green",
        histtype="step",
    )
    axs[2].hist(
        numu_cc_events[cat2],
        weights=numu_cc_events["w"],
        range=hist_range,
        bins=bins,
        color="blue",
        histtype="step",
    )
    axs[2].hist(
        nc_events[cat2],
        weights=nc_events["w"],
        range=hist_range,
        bins=bins,
        color="red",
        histtype="step",
    )
    axs[2].set_xlabel("Combined nc score", fontsize=17)
    axs[2].set_yscale("log")
    save(save_path)


def plot_curves(events, cat, save_path):
    """Plot the eff, pur, fom curves.

    Args:
        events (dict): events output
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(
        2, 2, figsize=(12, 12), gridspec_kw={"hspace": 0.2, "wspace": 0.3}
    )
    plt.setp(axs, xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.25, 0.5, 0.75, 1])
    styles = ["solid", "dashed", "dotted", "dashdot"]
    for i in range(len(events)):
        axs[0, 0].plot(
            events[i]["cuts"],
            events[i]["sig_effs"][cat],
            color="green",
            linestyle=styles[i],
            label="",
        )
        axs[0, 0].plot(
            events[i]["cuts"],
            events[i]["bkg_effs"][cat],
            color="red",
            linestyle=styles[i],
        )
    if cat == 0:
        axs[0, 0].set_xlabel(r"$\nu_{e}$ CC score", fontsize=22)
    elif cat == 1:
        axs[0, 0].set_xlabel(r"$\nu_{\mu}$ CC score", fontsize=22)
    axs[0, 0].set_ylabel("Efficiency", fontsize=22)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_xlim(0, 1)
    signal = Line2D(
        [0], [0], color="green", linewidth=2, linestyle="solid", label="signal"
    )
    bkg = Line2D(
        [0], [0], color="red", linewidth=2, linestyle="solid", label="background"
    )
    axs[0, 0].legend(handles=[signal, bkg], loc="center left")
    axs[0, 0].grid()

    for i in range(len(events)):
        axs[0, 1].plot(
            events[i]["cuts"],
            events[i]["purs"][cat],
            color="blue",
            linestyle=styles[i],
        )
        axs[0, 1].plot(
            events[i]["cuts"],
            events[i]["foms_0"][cat],
            color="black",
            linestyle=styles[i],
        )
    if cat == 0:
        axs[0, 1].set_xlabel(r"$\nu_{e}$ CC score", fontsize=22)
    elif cat == 1:
        axs[0, 1].set_xlabel(r"$\nu_{\mu}$ CC score", fontsize=22)
    axs[0, 1].set_ylabel("Purity or FOM", fontsize=22)
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].set_xlim(0, 1)
    pur = Line2D([0], [0], color="blue", linewidth=2, linestyle="solid", label="purity")
    fom = Line2D(
        [0], [0], color="black", linewidth=2, linestyle="solid", label="figure-of-merit"
    )
    axs[0, 1].legend(handles=[pur, fom], loc="center left")
    axs[0, 1].grid()

    for i in range(len(events)):
        axs[1, 0].plot(
            events[i]["bkg_effs"][cat],
            events[i]["sig_effs"][cat],
            color="black",
            linestyle=styles[i],
        )
    axs[1, 0].set_xlabel("Background efficiency", fontsize=22)
    axs[1, 0].set_ylabel("Signal efficiency", fontsize=22)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_xlim(0, 1)
    axs[1, 0].legend()
    axs[1, 0].grid()

    for i in range(len(events)):
        axs[1, 1].plot(
            events[i]["sig_effs"][cat],
            events[i]["purs"][cat],
            color="pink",
            linestyle=styles[i],
        )
    axs[1, 1].set_xlabel("Signal efficiency", fontsize=22)
    axs[1, 1].set_ylabel("Purity", fontsize=22)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_xlim(0, 1)
    axs[1, 1].legend()
    axs[1, 1].grid()

    save(save_path)


def plot_e_hists(events, ev, save_path):
    """Plot the eff and pur plot vs neutrino energy.

    Args:
        events (dict): events output
        ev (pd.DataFrame): events DataFrame
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(1.25, 8.25, 0.5)
    styles = ["solid", "dashed", "dotted", "dashdot"]

    for i in range(len(events)):
        axs[0].errorbar(
            bins,
            events[i]["fom_effs"][0][0][0],
            yerr=events[i]["fom_effs"][0][0][1],
            color="green",
            linestyle=styles[i],
        )
        axs[0].errorbar(
            bins,
            events[i]["fom_effs"][0][1][0],
            yerr=events[i]["fom_effs"][0][1][1],
            color="blue",
            linestyle=styles[i],
        )
        axs[0].errorbar(
            bins,
            events[i]["fom_effs"][0][2][0],
            yerr=events[i]["fom_effs"][0][2][1],
            color="red",
            linestyle=styles[i],
        )
        axs[0].errorbar(
            bins,
            events[i]["fom_purs"][0][0],
            yerr=events[i]["fom_purs"][0][1],
            color="black",
            linestyle=styles[i],
        )
    axs[0].hist(
        ev[ev["t_comb_cat"] == 0]["t_nu_energy"].to_numpy() / 1000,
        range=(1, 8),
        bins=14,
        color="green",
        density=True,
        alpha=0.3,
        weights=ev[ev["t_comb_cat"] == 0]["w"].to_numpy(),
    )
    axs[0].set_xlabel("Neutrino energy (GeV)", fontsize=22)
    axs[0].set_ylabel("Efficiency", fontsize=22)
    axs[0].set_ylim([0, 1])
    axs[0].set_title(r"$\nu_{e}$ CC signal")

    for i in range(len(events)):
        axs[1].errorbar(
            bins,
            events[i]["fom_effs"][1][0][0],
            yerr=events[i]["fom_effs"][1][0][1],
            color="green",
            linestyle=styles[i],
        )
        axs[1].errorbar(
            bins,
            events[i]["fom_effs"][1][1][0],
            yerr=events[i]["fom_effs"][1][1][1],
            color="blue",
            linestyle=styles[i],
        )
        axs[1].errorbar(
            bins,
            events[i]["fom_effs"][1][2][0],
            yerr=events[i]["fom_effs"][1][2][1],
            color="red",
            linestyle=styles[i],
        )
        axs[1].errorbar(
            bins,
            events[i]["fom_purs"][1][0],
            yerr=events[i]["fom_purs"][1][1],
            color="black",
            linestyle=styles[i],
        )
    axs[1].hist(
        ev[ev["t_comb_cat"] == 1]["t_nu_energy"].to_numpy() / 1000,
        range=(1, 8),
        bins=14,
        color="blue",
        density=True,
        alpha=0.3,
        weights=ev[ev["t_comb_cat"] == 1]["w"].to_numpy(),
    )
    axs[1].set_xlabel("Neutrino energy (GeV)", fontsize=22)
    axs[1].set_ylim([0, 1])
    axs[1].set_title(r"$\nu_{\mu}$ CC signal")
    save(save_path)


def plot_history(config, model, save_path, key="accuracy"):
    """Plot the training history of a list of models.

    Args:
        config (dotmap.DotMap): configuration namespace
        models (list[str]): list of models to plot
        save_path (str): path to save plot to
        key (str): Key to use for second metric alongside loss
    """
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    plt.setp(axs, xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    history = chipsnet.utils.model_history(config, model)
    epochs = np.arange(1, len(history["loss"]) + 1)
    axs.set_xlabel("epoch", fontsize=17)
    axs.set_ylabel(key, color="tab:red", fontsize=17)
    axs.plot(epochs, history[key], color="tab:red", linestyle="solid")
    axs.plot(epochs, history["val_" + key], color="tab:red", linestyle="dashed")
    axs.tick_params(axis="y", labelcolor="tab:red")
    axs_t = axs.twinx()  # instantiate a second axes that shares the same x-axis
    axs_t.set_ylabel(
        "loss", color="tab:blue", fontsize=17
    )  # we already handled the x-label with ax1
    axs_t.plot(epochs, history["loss"], color="tab:blue", linestyle="solid")
    axs_t.plot(epochs, history["val_loss"], color="tab:blue", linestyle="dashed")
    axs_t.tick_params(axis="y", labelcolor="tab:blue")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    save(save_path)
