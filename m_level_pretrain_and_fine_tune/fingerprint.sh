python main.py fingerprint --data_path exampledata/finetune/tox21.csv \
                           --features_path exampledata/finetune/tox21.npz \
                           --checkpoint_path /data/pj20/grover/finetune/tox21/fold_0/model_0/model.pt \
                           --fingerprint_source both \
                           --output /data/pj20/grover/fingerprints/fp_grover_tox21_both.npz


python main.py fingerprint --data_path exampledata/finetune/tox21.csv \
                           --features_path exampledata/finetune/tox21_fg_kge.npy \
                           --checkpoint_path /data/pj20/grover/finetune_kge/1200_fg/tox21/fold_1/model_0/model.pt \
                           --fingerprint_source both \
                           --output /data/pj20/grover/fingerprints/fp_kge_tox21_both.npz


python main.py fingerprint --data_path exampledata/finetune/tox21.csv \
                           --features_path exampledata/finetune/tox21_fg_kgnn.npy \
                           --checkpoint_path /data/pj20/grover/finetune_kge/1200_fg/tox21/fold_1/model_0/model.pt \
                           --fingerprint_source both \
                           --output /data/pj20/grover/fingerprints/fp_kgnn_tox21_both.npz