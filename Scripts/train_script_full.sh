# LCTEM 9x9, 3x3, 1x1
python3 ../main.py \
--common_path=../Experiment/LCTEM_9x9_denoising \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=9x9_blind_spot \
--model=9x9_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/LCTEM_3x3_denoising \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/LCTEM_1x1_denoising \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=1x1_blind_spot \
--model=1x1_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 


# clTEM 3x3, 1x1
python3 ../main.py \
--common_path=../Experiment/clTEM_3x3_denoising \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/clTEM_1x1_denoising \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=1x1_blind_spot \
--model=1x1_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 

# Pt_Ceria 3x3, 1x1
python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_3x3_denoising \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_1x1_denoising \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=1x1_blind_spot \
--model=1x1_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

# Au 3x3, 1x1
python3 ../main.py \
--common_path=../Experiment/Au_3x3_denoising \
--training_path=../Datasets/Au \
--gt_path=../Datasets/Au \
--data_path_test=../Datasets/Au \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=100 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

python3 ../main.py \
--common_path=../Experiment/Au_1x1_denoising \
--training_path=../Datasets/Au \
--gt_path=../Datasets/Au \
--data_path_test=../Datasets/Au \
--save_folder_name=experiment \
--version_folder_name=1x1_blind_spot \
--model=1x1_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=100 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

# MoS 3x3, 1x1
python3 ../main.py \
--common_path=../Experiment/MoS_3x3_denoising \
--training_path=../Datasets/MoS \
--gt_path=../Datasets/MoS \
--data_path_test=../Datasets/MoS \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

python3 ../main.py \
--common_path=../Experiment/MoS_1x1_denoising \
--training_path=../Datasets/MoS \
--gt_path=../Datasets/MoS \
--data_path_test=../Datasets/MoS \
--save_folder_name=experiment \
--version_folder_name=1x1_blind_spot \
--model=1x1_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

# Pd 3x3, 1x1
python3 ../main.py \
--common_path=../Experiment/Pd_3x3_denoising \
--training_path=../Datasets/Pd \
--gt_path=../Datasets/Pd \
--data_path_test=../Datasets/Pd \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

python3 ../main.py \
--common_path=../Experiment/Pd_1x1_denoising \
--training_path=../Datasets/Pd \
--gt_path=../Datasets/Pd \
--data_path_test=../Datasets/Pd \
--save_folder_name=experiment \
--version_folder_name=1x1_blind_spot \
--model=1x1_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

# STEM 10us, 5us 5x5, 3x3
python3 ../main.py \
--common_path=../Experiment/STEM10_5x5_denoising \
--training_path=../Datasets/STEM_10 \
--gt_path=../Datasets/STEM_10 \
--data_path_test=../Datasets/STEM_10 \
--save_folder_name=experiment \
--version_folder_name=5x5_blind_spot \
--model=5x5_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=50 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

python3 ../main.py \
--common_path=../Experiment/STEM5_5x5_denoising \
--training_path=../Datasets/STEM_5 \
--gt_path=../Datasets/STEM_5 \
--data_path_test=../Datasets/STEM_5 \
--save_folder_name=experiment \
--version_folder_name=5x5_blind_spot \
--model=5x5_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=50 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

python3 ../main.py \
--common_path=../Experiment/STEM10_3x3_denoising \
--training_path=../Datasets/STEM_10 \
--gt_path=../Datasets/STEM_10 \
--data_path_test=../Datasets/STEM_10 \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=50 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1

python3 ../main.py \
--common_path=../Experiment/STEM5_3x3_denoising \
--training_path=../Datasets/STEM_5 \
--gt_path=../Datasets/STEM_5 \
--data_path_test=../Datasets/STEM_5 \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=50 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1


# Cryo-ET 5x5
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
--batch_size=16 \
--max_epochs=100 \
--learning_rate=0.001 \
--subset_size=10 \
--loss_function='L2' \
--precision=16 \
--recursive_factor=1 \
--processor_num=20 \
--prepare_patch=1 \
--train=1 \
--test=1