#! /bin/bash

# Which training sample is best analysis? (sample)
python chipsnet/run.py analysis/config/train/sample/sample_both.yaml
python chipsnet/run.py analysis/config/train/sample/sample_uniform.yaml

# Which representation and channels are best analysis? (repr)
python chipsnet/run.py analysis/config/train/repr/repr_v_c.yaml
python chipsnet/run.py analysis/config/train/repr/repr_v_ct.yaml
python chipsnet/run.py analysis/config/train/repr/repr_o_ct.yaml
python chipsnet/run.py analysis/config/train/repr/repr_i_ct.yaml
python chipsnet/run.py analysis/config/train/repr/repr_v_cth.yaml
python chipsnet/run.py analysis/config/train/repr/repr_v_cth_stacked.yaml

# Which network architecture works best? (arch)
python chipsnet/run.py analysis/config/train/arch/arch_inception.yaml
python chipsnet/run.py analysis/config/train/arch/arch_resnet.yaml
python chipsnet/run.py analysis/config/train/arch/arch_inception_resnet.yaml

# Which categorisation is best analysis? (cat)
python chipsnet/run.py analysis/config/train/cat/cat_t_nc_comb_cat.yaml
python chipsnet/run.py analysis/config/train/cat/cat_t_comb_cat.yaml
python chipsnet/run.py analysis/config/train/cat/cat_split.yaml
python chipsnet/run.py analysis/config/train/cat/cat_split_learn.yaml

# Cosmic classification analysis (cosmic)
python chipsnet/run.py analysis/config/train/cosmic/cosmic.yaml
python chipsnet/run.py analysis/config/train/cosmic/cosmic_escapes.yaml
python chipsnet/run.py analysis/config/train/cosmic/cosmic_escapes_learn.yaml

# Beam classification analysis (beam)
python chipsnet/run.py analysis/config/train/beam/beam_primaries.yaml
python chipsnet/run.py analysis/config/train/beam/beam_primaries_learn.yaml

# Energy mutli analysis (energy_multi)
python chipsnet/run.py analysis/config/train/energy/energy_nu.yaml
python chipsnet/run.py analysis/config/train/energy/energy_lep.yaml
python chipsnet/run.py analysis/config/train/energy/energy_nu_lep.yaml
python chipsnet/run.py analysis/config/train/energy/energy_nu_lep_learn.yaml
python chipsnet/run.py analysis/config/train/energy/energy_nu_lep_vtx.yaml
python chipsnet/run.py analysis/config/train/energy/energy_nu_lep_vtx_learn.yaml

# Final models
python chipsnet/run.py analysis/config/train/final/final_cosmic.yaml
python chipsnet/run.py analysis/config/train/final/final_beam.yaml
python chipsnet/run.py analysis/config/train/final/final_nuel_ccdis_e.yaml
python chipsnet/run.py analysis/config/train/final/final_nuel_ccqel_e.yaml
python chipsnet/run.py analysis/config/train/final/final_nuel_ccres_e.yaml
python chipsnet/run.py analysis/config/train/final/final_nuel_cc_e.yaml
python chipsnet/run.py analysis/config/train/final/final_numu_ccdis_e.yaml
python chipsnet/run.py analysis/config/train/final/final_numu_ccqel_e.yaml
python chipsnet/run.py analysis/config/train/final/final_numu_ccres_e.yaml
python chipsnet/run.py analysis/config/train/final/final_numu_cc_e.yaml
python chipsnet/run.py analysis/config/train/final/final_nc_e.yaml

# Explanation model (stacked/no_reco)
python chipsnet/run.py analysis/config/train/explain/explain_cosmic.yaml
python chipsnet/run.py analysis/config/train/explain/explain_beam.yaml
python chipsnet/run.py analysis/config/train/explain/explain_energy.yaml

# Model studies (studies)
python chipsnet/run.py analysis/config/train/study/study_cosmic.yaml
python chipsnet/run.py analysis/config/train/study/study_beam.yaml
python chipsnet/run.py analysis/config/train/study/study_energy.yaml
