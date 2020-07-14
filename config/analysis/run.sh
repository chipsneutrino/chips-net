#! /bin/bash

# Which representation and channels are best analysis (repr)
#python chipsnet/run.py config/analysis/repr_v_c.yaml
#python chipsnet/run.py config/analysis/repr_v_ct.yaml
#python chipsnet/run.py config/analysis/repr_o_ct.yaml
#python chipsnet/run.py config/analysis/repr_i_ct.yaml
#python chipsnet/run.py config/analysis/repr_v_cth.yaml
#python chipsnet/run.py config/analysis/repr_v_cth_stacked.yaml

# Which training sample is best analysis (sample)
#python chipsnet/run.py config/analysis/sample_flux.yaml
#python chipsnet/run.py config/analysis/sample_uniform.yaml

# Which categorisation is best analysis (cat)
#python chipsnet/run.py config/analysis/cat_t_final_cat.yaml
#python chipsnet/run.py config/analysis/cat_t_all_cat.yaml
#python chipsnet/run.py config/analysis/cat_t_comb_cat.yaml
#python chipsnet/run.py config/analysis/cat_t_nc_cat.yaml
#python chipsnet/run.py config/analysis/cat_split.yaml
#python chipsnet/run.py config/analysis/cat_split_learn.yaml

# Cosmic classification analysis (cosmic)
#python chipsnet/run.py config/analysis/cosmic.yaml
#python chipsnet/run.py config/analysis/cosmic_vtx.yaml
#python chipsnet/run.py config/analysis/cosmic_vtx_learn.yaml

# Beam classification analysis (beam)
#python chipsnet/run.py config/analysis/beam_primaries.yaml
#python chipsnet/run.py config/analysis/beam_primaries_learn.yaml

# Energy estimation analysis (energy)
#python chipsnet/run.py config/analysis/energy.yaml
#python chipsnet/run.py config/analysis/lepton.yaml
#python chipsnet/run.py config/analysis/energy_lepton.yaml
#python chipsnet/run.py config/analysis/energy_lepton_learn.yaml

# Different sample energy estimations (energy)
#python chipsnet/run.py config/analysis/energy/nuel_cccoh_energy.yaml
#python chipsnet/run.py config/analysis/energy/nuel_ccdis_energy.yaml
#python chipsnet/run.py config/analysis/energy/nuel_ccqel_energy.yaml
#python chipsnet/run.py config/analysis/energy/nuel_ccres_energy.yaml
#python chipsnet/run.py config/analysis/energy/nuel_ccmec_energy.yaml
#python chipsnet/run.py config/analysis/energy/nuel_ccqelmec_energy.yaml
#python chipsnet/run.py config/analysis/energy/numu_cccoh_energy.yaml
#python chipsnet/run.py config/analysis/energy/numu_ccdis_energy.yaml
#python chipsnet/run.py config/analysis/energy/numu_ccqel_energy.yaml
#python chipsnet/run.py config/analysis/energy/numu_ccres_energy.yaml
#python chipsnet/run.py config/analysis/energy/numu_ccmec_energy.yaml
#python chipsnet/run.py config/analysis/energy/numu_ccqelmec_energy.yaml
#python chipsnet/run.py config/analysis/energy/numu_cc_energy.yaml
#python chipsnet/run.py config/analysis/energy/nc_energy.yaml

# Final beam model
python chipsnet/run.py config/analysis/final_cosmic.yaml
python chipsnet/run.py config/analysis/final_beam.yaml

# Explanation model (stacked/no_reco)
#python chipsnet/run.py config/analysis/explain.yaml

# Different height detector analysis (height)
#python chipsnet/run.py config/analysis/height_chips_1000.yaml
#python chipsnet/run.py config/analysis/height_chips_800.yaml
#python chipsnet/run.py config/analysis/height_chips_600.yaml
#python chipsnet/run.py config/analysis/height_chips_400.yaml