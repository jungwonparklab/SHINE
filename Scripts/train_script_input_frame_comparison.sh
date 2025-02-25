# Number of input frame comparison using simulated datasets

python3 ../main.py \
--common_path=../Experiment/LCTEM_9x9_denoising_1frame \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=LCTEM_9x9_denoising_1frame \
--model=9x9_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=1

python3 ../main.py \
--common_path=../Experiment/LCTEM_9x9_denoising_3frame \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=LCTEM_9x9_denoising_3frame \
--model=9x9_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=3

python3 ../main.py \
--common_path=../Experiment/LCTEM_9x9_denoising_5frame \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=LCTEM_9x9_denoising_5frame \
--model=9x9_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=5

python3 ../main.py \
--common_path=../Experiment/LCTEM_9x9_denoising_7frame \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=LCTEM_9x9_denoising_7frame \
--model=9x9_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=7

python3 ../main.py \
--common_path=../Experiment/LCTEM_9x9_denoising_9frame \
--training_path=../Datasets/LCTEM \
--gt_path=../Datasets/LCTEM_gt \
--data_path_test=../Datasets/LCTEM \
--save_folder_name=experiment \
--version_folder_name=LCTEM_9x9_denoising_9frame \
--model=9x9_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=9

python3 ../main.py \
--common_path=../Experiment/clTEM_3x3_denoising_1frame \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=clTEM_3x3_denoising_1frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=1

python3 ../main.py \
--common_path=../Experiment/clTEM_3x3_denoising_3frame \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=clTEM_3x3_denoising_3frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=3

python3 ../main.py \
--common_path=../Experiment/clTEM_3x3_denoising_5frame \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=clTEM_3x3_denoising_5frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=5

python3 ../main.py \
--common_path=../Experiment/clTEM_3x3_denoising_7frame \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=clTEM_3x3_denoising_7frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=7

python3 ../main.py \
--common_path=../Experiment/clTEM_3x3_denoising_9frame \
--training_path=../Datasets/clTEM \
--gt_path=../Datasets/clTEM_gt \
--data_path_test=../Datasets/clTEM \
--save_folder_name=experiment \
--version_folder_name=clTEM_3x3_denoising_9frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=9

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_3x3_denoising_1frame \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=Pt_Ceria_3x3_denoising_1frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=1

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_3x3_denoising_3frame \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=Pt_Ceria_3x3_denoising_3frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=3

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_3x3_denoising_5frame \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=Pt_Ceria_3x3_denoising_5frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=5

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_3x3_denoising_7frame \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=Pt_Ceria_3x3_denoising_7frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=7

python3 ../main.py \
--common_path=../Experiment/Pt_Ceria_3x3_denoising_9frame \
--training_path=../Datasets/Pt_Ceria \
--gt_path=../Datasets/Pt_Ceria_gt \
--data_path_test=../Datasets/Pt_Ceria \
--save_folder_name=experiment \
--version_folder_name=Pt_Ceria_3x3_denoising_9frame \
--model=3x3_blind \
--img_size=256 \
--batch_size=16 \
--max_epochs=200 \
--recursive_factor=10 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--test=1 \
--frame_num=9