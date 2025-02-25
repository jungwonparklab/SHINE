# Fig. 1H~K
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