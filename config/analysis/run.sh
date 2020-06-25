#! /bin/bash

python chipsnet/run.py config/analysis/beam_v_c.yaml
python chipsnet/run.py config/analysis/beam_v_ct.yaml
python chipsnet/run.py config/analysis/beam_o_ct.yaml
python chipsnet/run.py config/analysis/beam_i_ct.yaml
python chipsnet/run.py config/analysis/beam_v_cth.yaml

python chipsnet/run.py config/analysis/beam_v_cth_just_flux.yaml
python chipsnet/run.py config/analysis/beam_v_cth_just_uniform.yaml

python chipsnet/run.py config/analysis/t_all_cat_v_cth.yaml
python chipsnet/run.py config/analysis/t_comb_cat_v_cth.yaml
python chipsnet/run.py config/analysis/t_nc_cat_v_cth.yaml

python chipsnet/run.py config/analysis/beam_v_cth_resnet.yaml
python chipsnet/run.py config/analysis/beam_v_cth_inception.yaml

python chipsnet/run.py config/analysis/cosmic_v_ct.yaml
python chipsnet/run.py config/analysis/cosmic_o_ct.yaml
python chipsnet/run.py config/analysis/cosmic_i_ct.yaml

python chipsnet/run.py config/analysis/multi_beam_v_cth.yaml
python chipsnet/run.py config/analysis/multi_beam_cosmic_v_cth.yaml
python chipsnet/run.py config/analysis/multi_beam_energy_v_cth.yaml
python chipsnet/run.py config/analysis/multi_beam_vertex_v_cth.yaml