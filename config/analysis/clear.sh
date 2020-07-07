#! /bin/bash

# Which representation and channels are best analysis (repr)
rm -rf data/models/analysis/repr_v_c
rm -rf data/models/analysis/repr_v_ct
rm -rf data/models/analysis/repr_o_ct
rm -rf data/models/analysis/repr_i_ct
rm -rf data/models/analysis/repr_v_cth

# Which training sample is best analysis (sample)
rm -rf data/models/analysis/sample_flux
rm -rf data/models/analysis/sample_uniform

# Which categorisation is best analysis (cat)
rm -rf data/models/analysis/cat_t_final_cat
rm -rf data/models/analysis/cat_t_all_cat
rm -rf data/models/analysis/cat_t_comb_cat
rm -rf data/models/analysis/cat_t_nc_cat
rm -rf data/models/analysis/cat_split

# Cosmic classification analysis (cosmic)
rm -rf data/models/analysis/cosmic
rm -rf data/models/analysis/cosmic_vtx

# Explanation model (stacked/no_reco)
rm -rf data/models/analysis/explain

# Different height detector analysis (height)
rm -rf data/models/analysis/height_chips_1000
rm -rf data/models/analysis/height_chips_800
rm -rf data/models/analysis/height_chips_600
rm -rf data/models/analysis/height_chips_400

# Energy estimation analysis (energy)
rm -rf data/models/analysis/nuel_cccoh_energy
rm -rf data/models/analysis/nuel_ccdis_energy
rm -rf data/models/analysis/nuel_ccqel_energy
rm -rf data/models/analysis/nuel_ccres_energy
rm -rf data/models/analysis/nuel_ccmec_energy
rm -rf data/models/analysis/nuel_ccqelmec_energy
rm -rf data/models/analysis/numu_cccoh_energy
rm -rf data/models/analysis/numu_ccdis_energy
rm -rf data/models/analysis/numu_ccqel_energy
rm -rf data/models/analysis/numu_ccres_energy
rm -rf data/models/analysis/numu_ccmec_energy
rm -rf data/models/analysis/numu_ccqelmec_energy
rm -rf data/models/analysis/nuel_cc_energy
rm -rf data/models/analysis/numu_cc_energy
rm -rf data/models/analysis/nc_energy