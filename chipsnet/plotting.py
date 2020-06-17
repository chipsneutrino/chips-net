# -*- coding: utf-8 -*-

"""Utility module containing lots of helpful methods for evaluation and plotting."""

import numpy as np
import matplotlib.pyplot as plt

import chipsnet.utils


def save(name):
    plt.savefig("{}.pgf".format(name))
    plt.savefig("{}.pdf".format(name))
    plt.show()


def plot_cats(events_u, events_b, cat_map, save_path):
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
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={"hspace": 0.3})
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs[0, 0].imshow(
        images_dict["r_raw_charge_map_origin"][event], cmap="Reds", origin="lower"
    )
    axs[0, 0].set_title("$\phi$ and $\\theta$ from origin")
    axs[0, 0].set(xlabel="$\phi$ bins", ylabel="$\\theta$ bins")

    axs[0, 1].imshow(
        images_dict["r_raw_charge_map_iso"][event], cmap="Reds", origin="lower"
    )
    axs[0, 1].set_title("$x^{+}$ and $x^{-}$ from origin")
    axs[0, 1].set(xlabel="$x^{+}$ bins", ylabel="$x^{-}$ bins")

    axs[0, 2].imshow(
        images_dict["r_raw_charge_map_vtx"][event], cmap="Reds", origin="lower"
    )
    axs[0, 2].set_title("$\phi$ and $\\theta$ from vertex")
    axs[0, 2].set(xlabel="$\phi$ bins", ylabel="$\\theta$ bins")
    axs[0, 2].text(68, 3, "Desposited charge images", rotation=-90, fontsize=18)

    axs[1, 0].imshow(
        images_dict["r_raw_time_map_origin"][event], cmap="Reds", origin="lower"
    )
    axs[1, 0].set(xlabel="$\phi$ bins", ylabel="$\\theta$ bins")

    axs[1, 1].imshow(
        images_dict["r_raw_time_map_iso"][event], cmap="Reds", origin="lower"
    )
    axs[1, 1].set(xlabel="$x^{+}$ bins", ylabel="$x^{-}$ bins")

    axs[1, 2].imshow(
        images_dict["r_raw_time_map_vtx"][event], cmap="Reds", origin="lower"
    )
    axs[1, 2].set(xlabel="$\phi$ bins", ylabel="$\\theta$ bins")
    axs[1, 2].text(68, 10, "First hit time images", rotation=-90, fontsize=18)
    save("{}example_image_{}".format(save_path, event))


def plot_hough(images_dict, event, save_path):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), gridspec_kw={"hspace": 0.3})
    plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])
    axs.imshow(
        images_dict["r_raw_hit_hough_map_vtx"][event], cmap="Reds", origin="lower"
    )
    axs.set_title("$\phi$ and $\\theta$ from vertex")
    axs.set(xlabel="$\phi$ bins", ylabel="$\\theta$ bins")
    axs.text(68, 13, "Hough space image", rotation=-90, fontsize=18)
    save("{}example_hough_{}".format(save_path, event))


def plot_8bit_range(
    images_dict, max_charge=30, max_time=130, max_hough=4000, save_path=""
):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    # plt.setp(axs, xticks=[0, 16, 32, 48, 64], yticks=[0, 16, 32, 48, 64])

    hist_data = []
    for event in images_dict["r_raw_charge_map_vtx"]:
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
    for event in images_dict["r_raw_time_map_vtx"]:
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
    for event in images_dict["r_raw_hit_hough_map_vtx"]:
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


def plot_combined_values(events, prefix, save_name):
    bins = 25
    range = (0, 1)
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
        range=range,
        bins=bins,
        color="green",
        histtype="step",
    )
    axs[0].hist(
        numu_cc_events[cat0],
        weights=numu_cc_events["w"],
        range=range,
        bins=bins,
        color="blue",
        histtype="step",
    )
    axs[0].hist(
        nc_events[cat0],
        weights=nc_events["w"],
        range=range,
        bins=bins,
        color="red",
        histtype="step",
    )
    axs[0].set_xlabel("Combined nuel cc score", fontsize=17)
    axs[0].set_yscale("log")

    axs[1].hist(
        nuel_cc_events[cat1],
        weights=nuel_cc_events["w"],
        range=range,
        bins=bins,
        color="green",
        histtype="step",
    )
    axs[1].hist(
        numu_cc_events[cat1],
        weights=numu_cc_events["w"],
        range=range,
        bins=bins,
        color="blue",
        histtype="step",
    )
    axs[1].hist(
        nc_events[cat1],
        weights=nc_events["w"],
        range=range,
        bins=bins,
        color="red",
        histtype="step",
    )
    axs[1].set_xlabel("Combined numu cc score", fontsize=17)
    axs[1].set_yscale("log")

    axs[2].hist(
        nuel_cc_events[cat2],
        weights=nuel_cc_events["w"],
        range=range,
        bins=bins,
        color="green",
        histtype="step",
    )
    axs[2].hist(
        numu_cc_events[cat2],
        weights=numu_cc_events["w"],
        range=range,
        bins=bins,
        color="blue",
        histtype="step",
    )
    axs[2].hist(
        nc_events[cat2],
        weights=nc_events["w"],
        range=range,
        bins=bins,
        color="red",
        histtype="step",
    )
    axs[2].set_xlabel("Combined nc score", fontsize=17)
    axs[2].set_yscale("log")
    save(save_name)


def plot_curves(events, save_name):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    styles = ["solid", "dashed", "dotted", "dashdot"]
    for i in range(len(events)):
        axs[0].plot(
            events[i]["cuts"][0],
            events[i]["sig_effs"][0][0],
            color="green",
            linestyle=styles[i],
        )
        axs[0].plot(
            events[i]["cuts"][0],
            events[i]["bkg_effs"][0][0],
            color="red",
            linestyle=styles[i],
        )
    axs[0].set_xlabel("Combined nuel cc score cut", fontsize=17)
    axs[0].set_ylabel("Efficiency", fontsize=17)

    for i in range(len(events)):
        axs[1].plot(
            events[i]["cuts"][0],
            events[i]["purs"][0][0],
            color="blue",
            linestyle=styles[i],
        )
        axs[1].plot(
            events[i]["cuts"][0],
            events[i]["foms"][0][0],
            color="black",
            linestyle=styles[i],
        )
    axs[1].set_xlabel("Combined nuel cc score cut", fontsize=17)
    axs[1].set_ylabel("Purity or FOM", fontsize=17)

    for i in range(len(events)):
        axs[2].plot(
            events[i]["bkg_effs"][0][0],
            events[i]["sig_effs"][0][0],
            color="black",
            linestyle=styles[i],
        )
    axs[2].set_xlabel("Background efficiency", fontsize=17)
    axs[2].set_ylabel("Signal efficiency", fontsize=17)
    save(save_name)


def plot_e_hists(events, ev, save_name):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    bins = np.arange(1.25, 8.25, 0.5)
    styles = ["solid", "dashed", "dotted", "dashdot"]

    for i in range(len(events)):
        axs[0].errorbar(
            bins,
            events[i]["fom_effs"][0][0][0][0],
            yerr=events[i]["fom_effs"][0][0][0][1],
            color="green",
            linestyle=styles[i],
        )
        axs[0].errorbar(
            bins,
            events[i]["fom_effs"][0][0][1][0],
            yerr=events[i]["fom_effs"][0][0][1][1],
            color="blue",
            linestyle=styles[i],
        )
        axs[0].errorbar(
            bins,
            events[i]["fom_effs"][0][0][2][0],
            yerr=events[i]["fom_effs"][0][0][2][1],
            color="red",
            linestyle=styles[i],
        )
        axs[0].errorbar(
            bins,
            events[i]["fom_purs"][0][0][0],
            yerr=events[i]["fom_purs"][0][0][1],
            color="black",
            linestyle=styles[i],
        )
    axs[0].hist(
        ev[ev["t_comb_cat"] == 0]["t_nuEnergy"].to_numpy() / 1000,
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
            events[i]["fom_effs"][0][1][0][0],
            yerr=events[i]["fom_effs"][0][1][0][1],
            color="green",
            linestyle=styles[i],
        )
        axs[1].errorbar(
            bins,
            events[i]["fom_effs"][0][1][1][0],
            yerr=events[i]["fom_effs"][0][1][1][1],
            color="blue",
            linestyle=styles[i],
        )
        axs[1].errorbar(
            bins,
            events[i]["fom_effs"][0][1][2][0],
            yerr=events[i]["fom_effs"][0][1][2][1],
            color="red",
            linestyle=styles[i],
        )
        axs[1].errorbar(
            bins,
            events[i]["fom_purs"][0][1][0],
            yerr=events[i]["fom_purs"][0][1][1],
            color="black",
            linestyle=styles[i],
        )
    axs[1].hist(
        ev[ev["t_comb_cat"] == 1]["t_nuEnergy"].to_numpy() / 1000,
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
            events[i]["fom_effs"][0][2][0][0],
            yerr=events[i]["fom_effs"][0][2][0][1],
            color="green",
            linestyle=styles[i],
        )
        axs[2].errorbar(
            bins,
            events[i]["fom_effs"][0][2][1][0],
            yerr=events[i]["fom_effs"][0][2][1][1],
            color="blue",
            linestyle=styles[i],
        )
        axs[2].errorbar(
            bins,
            events[i]["fom_effs"][0][2][2][0],
            yerr=events[i]["fom_effs"][0][2][2][1],
            color="red",
            linestyle=styles[i],
        )
        axs[2].errorbar(
            bins,
            events[i]["fom_purs"][0][2][0],
            yerr=events[i]["fom_purs"][0][2][1],
            color="black",
            linestyle=styles[i],
        )
    axs[2].hist(
        ev[ev["t_comb_cat"] == 2]["t_nuEnergy"].to_numpy() / 1000,
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

    save(save_name)


def plot_history_comparison(config, models, save_name, key="accuracy"):
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
    save(save_name)
