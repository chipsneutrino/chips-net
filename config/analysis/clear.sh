#! /bin/bash

# Which training sample is best analysis (sample)
rm -rf data/models/analysis/sample/sample_both
rm -rf data/models/analysis/sample/sample_uniform

# Which representation and channels are best analysis (repr)
rm -rf data/models/analysis/repr/repr_v_c
rm -rf data/models/analysis/repr/repr_v_ct
rm -rf data/models/analysis/repr/repr_o_ct
rm -rf data/models/analysis/repr/repr_i_ct
rm -rf data/models/analysis/repr/repr_v_cth
rm -rf data/models/analysis/repr/repr_v_cth_stacked

# Which categorisation is best analysis (cat)
rm -rf data/models/analysis/cat/cat_t_final_cat
rm -rf data/models/analysis/cat/cat_t_all_cat
rm -rf data/models/analysis/cat/cat_t_comb_cat
rm -rf data/models/analysis/cat/cat_t_nc_cat
rm -rf data/models/analysis/cat/cat_split
rm -rf data/models/analysis/cat/cat_split_learn

# Cosmic classification analysis (cosmic)
rm -rf data/models/analysis/cosmic/cosmic
rm -rf data/models/analysis/cosmic/cosmic_vtx
rm -rf data/models/analysis/cosmic/cosmic_vtx_learn

# Beam classification analysis (beam)
rm -rf data/models/analysis/beam/beam_primaries
rm -rf data/models/analysis/beam/beam_primaries_learn

# Energy estimation analysis (energy)
rm -rf data/models/analysis/energy/energy_nu
rm -rf data/models/analysis/energy/energy_lep
rm -rf data/models/analysis/energy/energy_nu_lep
rm -rf data/models/analysis/energy/energy_nu_lep_learn
rm -rf data/models/analysis/energy/energy_nu_lep_vtx
rm -rf data/models/analysis/energy/energy_nu_lep_vtx_learn

# Final models
rm -rf data/models/analysis/final/final_cosmic
rm -rf data/models/analysis/final/final_beam
rm -rf data/models/analysis/final/final_nuel_cccoh_e
rm -rf data/models/analysis/final/final_nuel_ccdis_e
rm -rf data/models/analysis/final/final_nuel_ccqel_e
rm -rf data/models/analysis/final/final_nuel_ccres_e
rm -rf data/models/analysis/final/final_numu_cccoh_e
rm -rf data/models/analysis/final/final_numu_ccdis_e
rm -rf data/models/analysis/final/final_numu_ccqel_e
rm -rf data/models/analysis/final/final_numu_ccres_e
rm -rf data/models/analysis/final/final_numu_cc_e
rm -rf data/models/analysis/final/final_nc_e

# Explanation model (stacked/no_reco)
rm -rf data/models/analysis/explain/explain_cosmic
rm -rf data/models/analysis/explain/explain_beam
rm -rf data/models/analysis/explain/explain_energy

# Different height detector analysis (height)
rm -rf data/models/analysis/height/height_1000_cosmic
rm -rf data/models/analysis/height/height_1000_beam
rm -rf data/models/analysis/height/height_800_cosmic
rm -rf data/models/analysis/height/height_800_beam
rm -rf data/models/analysis/height/height_600_cosmic
rm -rf data/models/analysis/height/height_600_beam
rm -rf data/models/analysis/height/height_400_cosmic
rm -rf data/models/analysis/height/height_400_beam

# Light cone studies
rm -rf data/models/analysis/lcs/lcs_cosmic
rm -rf data/models/analysis/lcs/lcs_beam

# Model studies (studies)
rm -rf data/models/analysis/study/study_cosmic
rm -rf data/models/analysis/study/study_beam
rm -rf data/models/analysis/study/study_energy