# -*- coding: utf-8 -*-

"""Plotting module containing lots of plotting methods for the analysis notebook."""

import numpy as np
import matplotlib.pyplot as plt

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
    width = 0.45

    fig, axs = plt.subplots(1, 1, figsize=(18, 5), gridspec_kw={"hspace": 0.3})
    axs.bar(
        cats + width / 2, data_u, color="royalblue", width=width, label="uniform sample"
    )
    axs.bar(
        cats - width / 2, data_b, color="firebrick", width=width, label="flux sample"
    )
    axs.set_xticks(cats)
    axs.set_xticklabels(cat_map["labels"], fontsize=14, rotation="vertical")
    axs.set_ylabel("Testing events")
    axs.legend()
    save(save_path + cat_map["name"])


def plot_hit_time(images_dict, event, save_path):
    """Plot hit and time channels for the different image representations.

    Args:
        images_dict (dict): images dictionary
        event (int): event to use
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={"hspace": 0.3})
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs[0, 0].imshow(
        images_dict["r_charge_map_origin"][event], cmap="Reds", origin="lower"
    )
    axs[0, 0].set_title(r"$\phi$ and $\theta$ from origin")
    axs[0, 0].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")

    axs[0, 1].imshow(
        images_dict["r_charge_map_iso"][event], cmap="Reds", origin="lower"
    )
    axs[0, 1].set_title(r"$x^{+}$ and $x^{-}$ from origin")
    axs[0, 1].set(xlabel=r"$x^{+}$ bins", ylabel=r"$x^{-}$ bins")

    axs[0, 2].imshow(
        images_dict["r_charge_map_vtx"][event], cmap="Reds", origin="lower"
    )
    axs[0, 2].set_title(r"$\phi$ and $\theta$ from vertex")
    axs[0, 2].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs[0, 2].text(68, 3, "Desposited charge images", rotation=-90, fontsize=18)

    axs[1, 0].imshow(
        images_dict["r_time_map_origin"][event], cmap="Reds", origin="lower"
    )
    axs[1, 0].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")

    axs[1, 1].imshow(images_dict["r_time_map_iso"][event], cmap="Reds", origin="lower")
    axs[1, 1].set(xlabel=r"$x^{+}$ bins", ylabel=r"$x^{-}$ bins")

    axs[1, 2].imshow(images_dict["r_time_map_vtx"][event], cmap="Reds", origin="lower")
    axs[1, 2].set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs[1, 2].text(68, 10, "First hit time images", rotation=-90, fontsize=18)
    save("{}example_image_{}".format(save_path, event))


def plot_hough(images_dict, event, save_path):
    """Plot the hough channel representation.

    Args:
        images_dict (dict): images dictionary
        event (int): event to use
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), gridspec_kw={"hspace": 0.3})
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs.imshow(images_dict["r_hough_map_vtx"][event], cmap="Reds", origin="lower")
    axs.set_title(r"$\phi$ and $\theta$ from vertex")
    axs.set(xlabel=r"$\phi$ bins", ylabel=r"$\theta$ bins")
    axs.text(68, 13, "Hough space image", rotation=-90, fontsize=18)
    save("{}example_hough_{}".format(save_path, event))


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
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    # plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])

    hist_data = []
    for event in images_dict["r_charge_map_vtx"]:
        hist_data.append(event.reshape(4096))
    hist_data = np.concatenate(hist_data, axis=0)
    occurrences = np.count_nonzero(hist_data == 255) / len(hist_data)
    axs[0].hist(hist_data, range=(1, 256), bins=255, color="grey", histtype="step")
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
    axs[1].hist(hist_data, range=(1, 256), bins=255, color="grey", histtype="step")
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
    axs[2].hist(hist_data, range=(1, 256), bins=255, color="grey", histtype="step")
    axs[2].set_title(
        "[0,{}], outside range: {:.4f}".format(max_hough, occurrences), fontsize=17
    )
    axs[2].set(xlabel="Hough 8-bit value", ylabel="Frequency")
    axs[2].label_outer()

    save(save_path + "8_bit_range")


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


def plot_curves(events, save_path):
    """Plot the eff, pur, fom curves.

    Args:
        events (dict): events output
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    styles = ["solid", "dashed", "dotted", "dashdot"]
    for i in range(len(events)):
        axs[0].plot(
            events[i]["cuts"],
            events[i]["sig_effs"][0],
            color="green",
            linestyle=styles[i],
        )
        axs[0].plot(
            events[i]["cuts"],
            events[i]["bkg_effs"][0],
            color="red",
            linestyle=styles[i],
        )
    axs[0].set_xlabel("Combined nuel cc score cut", fontsize=17)
    axs[0].set_ylabel("Efficiency", fontsize=17)

    for i in range(len(events)):
        axs[1].plot(
            events[i]["cuts"], events[i]["purs"][0], color="blue", linestyle=styles[i],
        )
        axs[1].plot(
            events[i]["cuts"], events[i]["foms"][0], color="black", linestyle=styles[i],
        )
    axs[1].set_xlabel("Combined nuel cc score cut", fontsize=17)
    axs[1].set_ylabel("Purity or FOM", fontsize=17)

    for i in range(len(events)):
        axs[2].plot(
            events[i]["bkg_effs"][0],
            events[i]["sig_effs"][0],
            color="black",
            linestyle=styles[i],
        )
    axs[2].set_xlabel("Background efficiency", fontsize=17)
    axs[2].set_ylabel("Signal efficiency", fontsize=17)
    save(save_path)


def plot_e_hists(events, ev, save_path):
    """Plot the eff and pur plot vs neutrino energy.

    Args:
        events (dict): events output
        ev (pd.DataFrame): events DataFrame
        save_path (str): path to save plot to
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
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
        alpha=0.5,
        weights=ev[ev["t_comb_cat"] == 0]["w"].to_numpy(),
    )
    axs[0].set_xlabel("Neutrino energy (GeV)", fontsize=17)
    axs[0].set_ylabel("Efficiency", fontsize=17)
    axs[0].set_ylim([0, 1])
    axs[0].set_title("nuel-cc signal")

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
        alpha=0.5,
        weights=ev[ev["t_comb_cat"] == 1]["w"].to_numpy(),
    )
    axs[1].set_xlabel("Neutrino energy (GeV)", fontsize=17)
    axs[1].set_ylim([0, 1])
    axs[1].set_title("numu-cc signal")

    for i in range(len(events)):
        axs[2].errorbar(
            bins,
            events[i]["fom_effs"][2][0][0],
            yerr=events[i]["fom_effs"][2][0][1],
            color="green",
            linestyle=styles[i],
        )
        axs[2].errorbar(
            bins,
            events[i]["fom_effs"][2][1][0],
            yerr=events[i]["fom_effs"][2][1][1],
            color="blue",
            linestyle=styles[i],
        )
        axs[2].errorbar(
            bins,
            events[i]["fom_effs"][2][2][0],
            yerr=events[i]["fom_effs"][2][2][1],
            color="red",
            linestyle=styles[i],
        )
        axs[2].errorbar(
            bins,
            events[i]["fom_purs"][2][0],
            yerr=events[i]["fom_purs"][2][1],
            color="black",
            linestyle=styles[i],
        )
    axs[2].hist(
        ev[ev["t_comb_cat"] == 2]["t_nu_energy"].to_numpy() / 1000,
        range=(1, 8),
        bins=14,
        color="red",
        density=True,
        alpha=0.5,
        weights=ev[ev["t_comb_cat"] == 2]["w"].to_numpy(),
    )
    axs[2].set_xlabel("Neutrino energy (GeV)", fontsize=17)
    axs[2].set_ylim([0, 1])
    axs[2].set_title("nc signal")

    save(save_path)


def plot_history_comparison(config, models, save_path, key="accuracy"):
    """Plot the training history of a list of models.

    Args:
        config (dotmap.DotMap): configuration namespace
        models (list[str]): list of models to plot
        save_path (str): path to save plot to
        key (str): Key to use for second metric alongside loss
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for i, model in enumerate(models):
        history = chipsnet.utils.model_history(config, model)
        epochs = np.arange(1, len(history["loss"]) + 1)
        axs[i].set_xlabel("epoch", fontsize=17)
        axs[i].set_ylabel(key, color="tab:red", fontsize=17)
        axs[i].plot(epochs, history[key], color="tab:red", linestyle="solid")
        axs[i].plot(epochs, history["val_" + key], color="tab:red", linestyle="dashed")
        axs[i].tick_params(axis="y", labelcolor="tab:red")
        axs_t = axs[i].twinx()  # instantiate a second axes that shares the same x-axis
        axs_t.set_ylabel(
            "loss", color="tab:blue", fontsize=17
        )  # we already handled the x-label with ax1
        axs_t.plot(epochs, history["loss"], color="tab:blue", linestyle="solid")
        axs_t.plot(epochs, history["val_loss"], color="tab:blue", linestyle="dashed")
        axs_t.tick_params(axis="y", labelcolor="tab:blue")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    save(save_path)


def plot_true_vs_reco_e(events, axs, key="t_nu_energy"):
    axs.hist2d(
        events["t_nu_energy"],
        e_ev[e_ev["t_comb_cat"] == 0]["energy_pred_t_nu_energy"],
        range=e_range,
        bins=e_bins,
        weights=e_ev[e_ev["t_comb_cat"] == 0]["w"],
        cmap="Reds",
    )
    axs.grid()
    axs.label_outer()
    axs.set(xlabel=r"True energy (MeV)", ylabel=r"Reco energy (MeV)")
    axs.set_title(r"Nuel CC")
