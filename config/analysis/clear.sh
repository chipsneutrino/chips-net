#! /bin/bash

# Which representation and channels are best analysis (repr)
rm -rf data/models/analysis/repr_v_c.yaml
rm -rf data/models/analysis/repr_v_ct.yaml
rm -rf data/models/analysis/repr_o_ct.yaml
rm -rf data/models/analysis/repr_i_ct.yaml
rm -rf data/models/analysis/repr_v_cth.yaml
rm -rf data/models/analysis/repr_v_cth_stacked.yaml

# Which training sample is best analysis (sample)
rm -rf data/models/analysis/sample_flux.yaml
rm -rf data/models/analysis/sample_uniform.yaml

# Which categorisation is best analysis (cat)
rm -rf data/models/analysis/cat_t_final_cat.yaml
rm -rf data/models/analysis/cat_t_all_cat.yaml
rm -rf data/models/analysis/cat_t_comb_cat.yaml
rm -rf data/models/analysis/cat_t_nc_cat.yaml
rm -rf data/models/analysis/cat_split.yaml
rm -rf data/models/analysis/cat_split_learn.yaml

# Cosmic classification analysis (cosmic)
rm -rf data/models/analysis/cosmic.yaml
rm -rf data/models/analysis/cosmic_vtx.yaml
rm -rf data/models/analysis/cosmic_vtx_learn.yaml

# Beam classification analysis (beam)
rm -rf data/models/analysis/beam_primaries.yaml
rm -rf data/models/analysis/beam_primaries_learn.yaml

# Energy estimation analysis (energy)
rm -rf data/models/analysis/energy.yaml
rm -rf data/models/analysis/lepton.yaml
rm -rf data/models/analysis/energy_lepton.yaml
rm -rf data/models/analysis/energy_lepton_learn.yaml

# Different sample energy estimations (energy)
rm -rf data/models/analysis/nuel_cccoh_energy.yaml
rm -rf data/models/analysis/nuel_ccdis_energy.yaml
rm -rf data/models/analysis/nuel_ccqel_energy.yaml
rm -rf data/models/analysis/nuel_ccres_energy.yaml
rm -rf data/models/analysis/nuel_ccmec_energy.yaml
rm -rf data/models/analysis/nuel_ccqelmec_energy.yaml
rm -rf data/models/analysis/numu_cccoh_energy.yaml
rm -rf data/models/analysis/numu_ccdis_energy.yaml
rm -rf data/models/analysis/numu_ccqel_energy.yaml
rm -rf data/models/analysis/numu_ccres_energy.yaml
rm -rf data/models/analysis/numu_ccmec_energy.yaml
rm -rf data/models/analysis/numu_ccqelmec_energy.yaml
rm -rf data/models/analysis/numu_cc_energy.yaml
rm -rf data/models/analysis/nc_energy.yaml

# Final beam model
rm -rf data/models/analysis/beam_final.yaml
rm -rf data/models/analysis/beam_final_learn.yaml

# Explanation model (stacked/no_reco)
rm -rf data/models/analysis/explain.yaml

# Different height detector analysis (height)
rm -rf data/models/analysis/height_chips_1000.yaml
rm -rf data/models/analysis/height_chips_800.yaml
rm -rf data/models/analysis/height_chips_600.yaml
rm -rf data/models/analysis/height_chips_400.yaml