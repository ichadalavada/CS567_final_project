Dataset information
Kitti Dataset: https://docs.ultralytics.com/datasets/detect/kitti
Scooter Dataset: https://www.kaggle.com/datasets/trainingdatapro/electric-scooters-tracking
Caltech Dataset: https://www.kaggle.com/datasets/abhinavsasikumar/caltech-pedestrian-yolo?resource=download
IDD Dataset: https://idd.insaan.iiit.ac.in/dataset/download/


1. Download the datasets and store in respective folders
2. Run preprocess_to_pickle.py on kitti, once, scooter_data, and IDD datasets (make a IDD folder with the test set for IDD).
3. Run merge_idd_data.py on all three datasets as well by changing the file paths in the file
4. Run split_idd_remainder.py to create a separate test from the data that is not merged into the IDD dataset
5. run caltech_yolo/merge_idd_data to add the caltech data to the once dataset pickle files
6. Then run train_and_evaluate_merged.py in each of the folders to get individual model accuracies and generate keras files
7. Run test_idd_remainder.py --model <path_to_keras_file> to find the model accuracy on the test set
8. Modify the file paths in ensemble_model/boosting_model to link to the keras files that you want to train XGBoost and AdaBoost on
9. Modify the file paths in ensemble_model/ensemble_model to link to the keras files that you want to train bagging methods on
