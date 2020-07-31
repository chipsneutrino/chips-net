#! /bin/bash

# Which representation and channels are best analysis (repr)
python chipsnet/run.py config/analysis/repr/repr_v_c.yaml
python chipsnet/run.py config/analysis/repr/repr_v_ct.yaml
python chipsnet/run.py config/analysis/repr/repr_o_ct.yaml
python chipsnet/run.py config/analysis/repr/repr_i_ct.yaml
python chipsnet/run.py config/analysis/repr/repr_v_cth.yaml
python chipsnet/run.py config/analysis/repr/repr_v_cth_stacked.yaml

# Which training sample is best analysis (sample)
python chipsnet/run.py config/analysis/sample/sample_flux.yaml
python chipsnet/run.py config/analysis/sample/sample_uniform.yaml

# Which categorisation is best analysis (cat)
python chipsnet/run.py config/analysis/cat/cat_t_final_cat.yaml
python chipsnet/run.py config/analysis/cat/cat_t_all_cat.yaml
python chipsnet/run.py config/analysis/cat/cat_t_comb_cat.yaml
python chipsnet/run.py config/analysis/cat/cat_t_nc_cat.yaml
python chipsnet/run.py config/analysis/cat/cat_split.yaml
python chipsnet/run.py config/analysis/cat/cat_split_learn.yaml

# Cosmic classification analysis (cosmic)
python chipsnet/run.py config/analysis/cosmic/cosmic.yaml
python chipsnet/run.py config/analysis/cosmic/cosmic_vtx.yaml
python chipsnet/run.py config/analysis/cosmic/cosmic_vtx_learn.yaml

# Beam classification analysis (beam)
python chipsnet/run.py config/analysis/beam/beam_primaries.yaml
python chipsnet/run.py config/analysis/beam/beam_primaries_learn.yaml

# Energy estimation analysis (energy)
python chipsnet/run.py config/analysis/energy/energy_nu.yaml
python chipsnet/run.py config/analysis/energy/energy_lep.yaml
python chipsnet/run.py config/analysis/energy/energy_nu_lep.yaml
python chipsnet/run.py config/analysis/energy/energy_nu_lep_learn.yaml
python chipsnet/run.py config/analysis/energy/energy_nu_lep_vtx.yaml
python chipsnet/run.py config/analysis/energy/energy_nu_lep_vtx_learn.yaml

# Final models
python chipsnet/run.py config/analysis/final/final_cosmic.yaml
python chipsnet/run.py config/analysis/final/final_beam.yaml
python chipsnet/run.py config/analysis/final/final_nuel_cccoh_e.yaml
python chipsnet/run.py config/analysis/final/final_nuel_ccdis_e.yaml
python chipsnet/run.py config/analysis/final/final_nuel_ccqel_e.yaml
python chipsnet/run.py config/analysis/final/final_nuel_ccres_e.yaml
python chipsnet/run.py config/analysis/final/final_numu_cccoh_e.yaml
python chipsnet/run.py config/analysis/final/final_numu_ccdis_e.yaml
python chipsnet/run.py config/analysis/final/final_numu_ccqel_e.yaml
python chipsnet/run.py config/analysis/final/final_numu_ccres_e.yaml
python chipsnet/run.py config/analysis/final/final_numu_cc_e.yaml
python chipsnet/run.py config/analysis/final/final_nc_e.yaml

# Explanation model (stacked/no_reco)
python chipsnet/run.py config/analysis/explain/explain_cosmic.yaml
python chipsnet/run.py config/analysis/explain/explain_beam.yaml
python chipsnet/run.py config/analysis/explain/explain_energy.yaml

# Different height detector analysis (height)
python chipsnet/run.py config/analysis/height/height_1000_cosmic.yaml
python chipsnet/run.py config/analysis/height/height_1000_beam.yaml
python chipsnet/run.py config/analysis/height/height_800_cosmic.yaml
python chipsnet/run.py config/analysis/height/height_800_beam.yaml
python chipsnet/run.py config/analysis/height/height_600_cosmic.yaml
python chipsnet/run.py config/analysis/height/height_600_beam.yaml
python chipsnet/run.py config/analysis/height/height_400_cosmic.yaml
python chipsnet/run.py config/analysis/height/height_400_beam.yaml

# Light cone studies
python chipsnet/run.py config/analysis/lcs/lcs_cosmic.yaml
python chipsnet/run.py config/analysis/lcs/lcs_beam.yaml

# Model studies (studies)
python chipsnet/run.py config/analysis/study/study_cosmic.yaml
python chipsnet/run.py config/analysis/study/study_beam.yaml
python chipsnet/run.py config/analysis/study/study_energy.yaml