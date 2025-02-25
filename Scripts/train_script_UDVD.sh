# UDVD, UDVD_e

python3 ../main.py \
--common_path=../Experiment/LCTEM_UDVD_denoising \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=UDVD \
--model=UDVD \
--img_size=256 \
--batch_size=2 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=32 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/clTEM_UDVD_denoising \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=UDVD \
--model=UDVD \
--img_size=256 \
--batch_size=2 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=32 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_UDVD_denoising \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=UDVD \
--model=UDVD \
--img_size=256 \
--batch_size=2 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=32 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/LCTEM_UDVD_Extended_denoising \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=UDVD_e \
--model=UDVD_e \
--img_size=256 \
--batch_size=2 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=32 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/clTEM_UDVD_Extended_denoising \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=UDVD_e \
--model=UDVD_e \
--img_size=256 \
--batch_size=2 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=32 \
--loss_function='L2' \
--test=1 

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_UDVD_Extended_denoising \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=UDVD_e \
--model=UDVD_e \
--img_size=256 \
--batch_size=2 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=32 \
--loss_function='L2' \
--test=1