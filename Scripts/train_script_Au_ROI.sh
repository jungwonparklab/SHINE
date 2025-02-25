# Fig. S3.
python3 ../main.py \
--common_path=../Experiment/Au_3x3_denoising_ROI1 \
--training_path=../Datasets/Au_crop1 \
--gt_path=../Datasets/Au_crop1 \
--data_path_test=../Datasets/Au_crop1 \
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
--test=1 \
--gpus=1

python3 ../main.py \
--common_path=../Experiment/Au_3x3_denoising_ROI2 \
--training_path=../Datasets/Au_crop2 \
--gt_path=../Datasets/Au_crop2 \
--data_path_test=../Datasets/Au_crop2 \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=100 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--gpus=1

