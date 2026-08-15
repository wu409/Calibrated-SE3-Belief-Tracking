Step1 Generating corruption datasets:
'''
python 0-corruption.py --dataset_base ./datasets/YCBInEOAT --out_dir ./datasets/YCBInEOAT_Corrupted --sequences mustard0 bleach_hard_00_03_chaitanya bleach0 --occlusion_rate 0.4 0.6 --dropout_rate 0.6
'''


Step2: Run SE(3)TrackNet predition.py to generates predictions in ./results/bleach0/, using the datasets you want to predict:
'''
python predict.py ^ --mode ycbineoat --YCBInEOAT_dir datasets\YCBInEOAT\bleach_hard_00_03_chaitanya --train_data_path datasets\YCBInEOAT_data\bleach_cleanser\train_data_blender_DR --ckpt_dir YCBInEOAT_weights\bleach_cleanser\model_best_val.pth.tar --mean_std_path YCBInEOAT_weights\bleach_cleanser --class_id 12 --model_path datasets\YCB_Video_Models\CADmodels\021_bleach_cleanser\textured.obj --outdir results/bleach_hard_00_03_chaitanya_black10
'''

Step 3: Evaluating ADD / ADD-S AUC metrics on prediction 
'''
python eval_ycbineoat.py --YCBInEOAT_dir ./datasets/YCBInEOAT --class_id 12 --ycb_dir ./datasets/YCB_Video_Models/ --res_dir ./results/
'''


Step4:  Extracting reliability feature and  labeling
'''
python 1-risk_label.py ^
--ycb_dir ./datasets/YCBInEOAT ^
--data_dir ./datasets/YCBInEOAT_Corrupted ^
--res_dir ./results_collection --mesh_path_root ./datasets/YCB_Video_Models/CADmodels ^
--target_seqs mustard0 bleach_hard_00_03_chaitanya bleach0 ^
--corruption_lists _occ40 _black10 _clean _drop60 _occ60 ^
--ci_object "bleach_hard_00_03_chaitanya" ^
--ci_episode _black10_2 _black10_3 _black10_4 _black10_5 
--cad_models_seq 006_mustard_bottle 021_bleach_cleanser 021_bleach_cleanser 021_bleach_cleanser ^
--risk_threshold 0.5
'''


Step5: Train and evaluation
'''
python 2-train_evaluation.py ^
--csv_path ./per_frame_label_threshold0.5.csv ^
--result_dir "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10" "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_2" "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_3" "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_4" "./results_collection/bleach_hard_00_03_chaitanya/bleach_hard_00_03_chaitanya_black10_5" ^
--gt_dir ./datasets/YCBInEOAT/bleach_hard_00_03_chaitanya/annotated_poses ^
--point_path ./datasets/YCB_Video_Models/CADmodels/021_bleach_cleanser/points.xyz ^
--train_seqs bleach0 mustard0 ^
--test_base_seq bleach_hard_00_03_chaitanya ^
--data_dir ./datasets/YCBInEOAT_Corrupted ^
--p_help_threshold 0.50 ^
--alpha 0.5 ^
--risk_threshold 0.5
'''




