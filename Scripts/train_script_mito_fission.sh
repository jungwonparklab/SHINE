# Fig. 6.
python3 ../main.py \
--file_type='mrc' \
--common_path=../Experiment/Tomo_mito_fission \
--training_path=../Datasets/Mito_fission \
--patches_folder=../Datasets/Mito_fission_patches/ \
--data_path_test=../Datasets/Mito_fission \
--patch_ratio=0.5 \
--patch_size=1024 \
--patch_stride=768 \
--save_folder_name=experiment \
--version_folder_name=5x5_blind_spot \
--model=5x5_blind  \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--learning_rate=0.001 \
--subset_size=10 \
--loss_function='L2' \
--precision=16 \
--recursive_factor=1 \
--processor_num=20 \
--prepare_patch=1 \
--train=1 \
--test=1 \
--gpus=2