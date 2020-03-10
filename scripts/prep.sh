#! /bin/bash

python preprocess.py /unix/chips/production/beam_nuel_all/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_coh_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_coh_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_dis_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_dis_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_qel_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_qel_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_res_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_res_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_nuel_pizero_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_all/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_coh_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_coh_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_dis_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_dis_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_qel_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_qel_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_res_cc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_res_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/beam_numu_pizero_nc/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/cosmic_numu_all/ -g chips_1200_sk1pe --all --reduce
python preprocess.py /unix/chips/production/cosmic_numu_beam_dir/ -g chips_1200_sk1pe --all --reduce


