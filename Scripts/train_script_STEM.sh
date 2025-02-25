# Fig. 5.
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
