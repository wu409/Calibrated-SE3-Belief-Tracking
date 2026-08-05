Step1 Generating corruption datasets:
'''
python 0-corruption.py
'''
There are some parameters can be choosen.

Step2: Run SE(3)TrackNet predition.py to generates predictions in ./results/bleach0/, using the datasets you want to predict:
'''
python predict.py --mode ycbineoat --YCBInEOAT_dir datasets\YCBInEOAT\bleach_hard_00_03_chaitanya --train_data_path datasets\YCBInEOAT_data\bleach_cleanser\train_data_blender_DR --ckpt_dir YCBInEOAT_weights\bleach_cleanser\model_best_val.pth.tar --mean_std_path YCBInEOAT_weights\bleach_cleanser --class_id 12 --model_path datasets\YCB_Video_Models\CADmodels\021_bleach_cleanser\textured.obj --outdir results/bleach_hard_00_03_chaitanya
'''

Step 3: Evaluating ADD / ADD-S AUC metrics on prediction 
'''
python eval_ycbineoat.py --YCBInEOAT_dir ./datasets/YCBInEOAT --class_id 12 --ycb_dir ./datasets/YCB_Video_Models/ --res_dir ./results/
'''


Step4:  Extracting reliability feature and  labeling
'''python 1-harm_label.py
'''


Step5: Analysing the harm predictor
'''
python 2-risk_model_train.py
'''

Step 6: Experimental Verification
'''
python 3-evaluation.py
'''



