#! /bin/bash

run ./config/analysis/t_all_cat_v_c_uniform.yaml
run ./config/analysis/t_all_cat_v_ct_uniform.yaml
run ./config/analysis/t_all_cat_o_ct_uniform.yaml
run ./config/analysis/t_all_cat_i_ct_uniform.yaml
run ./config/analysis/t_all_cat_v_cth_uniform.yaml
run ./config/analysis/t_all_cat_v_cth_flux.yaml

run ./config/analysis/t_cosmic_cat_v_cth_uniform.yaml
run ./config/analysis/t_cosmic_cat_v_cth_flux.yaml

run ./config/analysis/multi_simple_v_cth_uniform.yaml
run ./config/analysis/multi_simple_v_cth_flux.yaml
run ./config/analysis/multi_simple_energy_v_cth_uniform.yaml