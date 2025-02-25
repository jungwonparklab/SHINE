# N2V

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_N2V_denoising \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=N2V \
--model=N2V \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--frame_num=1 \
--loss_function='L2' \
--test=1 