import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
from Preprocess import Preprocess
from EDA_train import EDA_train
from EDA_test import EDA_test
from Feature_Selection_1 import FeatureSelection
# from Feature_Selection_2 import FeatureSelection2
from processing2 import Processing

def get_all_files_in_path(path, prefixes=["test", "train"]):
    folder_path = Path(path)
    return [file for prefix in prefixes for file in folder_path.glob(f"{prefix}*")]

def process_files_for_eda_and_features(input_dir, output_dir, class_type):
    instance = class_type(input_dir, output_dir)
    instance.plot_and_save_sensors()
    print(f'\033[91m{class_type.__name__} is completed\033[0m')

def main():
    data_root_inputs = Path("/Users/milad/WQD7002/data/data2008")
    data_root = "/Users/milad/WQD7002/data/data2008"
    main_dir_dataset = f"{data_root}"
    pre_processed_data_dir = f"{data_root}/Pre-Processed"
    eda_output_dir = f"{data_root}/EDA"
    feature_output_dir_1 = f"{data_root}/SelectedFeature(1)"
    feature_output_dir_2 = f"{data_root}/SelectedFeature(2)"
    processing_input_dir_1 = Path("/Users/milad/WQD7002/data/data2008/Pre-Processed")
    processing_input_dir_2 = Path("/Users/milad/WQD7002/data/data2008/Pre-Processed")
    processed_output_dir_1 = processing_input_dir_1 / "Processed(1)"
    processed_output_dir_2 = processing_input_dir_2 / "Processed(2)"
    train_path = Path("/Users/milad/WQD7002/data/data2008/Pre-Processed/Processed(1)/train_FD001_cleaned_Processed.csv")
    test_path = Path("/Users/milad/WQD7002/data/data2008/Pre-Processed/Processed(1)/train_FD001_cleaned_Processed.csv")
    actual_RUL_path = Path("/Users/milad/WQD7002/data/data2008")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
    # Preprocessing
    all_files = get_all_files_in_path(main_dir_dataset)
    for input_path in all_files:
        print('\033[92m' + f"Processing: {input_path}" + '\033[0m')
        preprocess = Preprocess(str(input_path), pre_processed_data_dir)
        preprocess.default_preprocess()
        print('\033[94m' + f"Output saved for: {input_path}" + '\033[0m')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # EDA
    # process_files_for_eda_and_features(pre_processed_data_dir, f"{eda_output_dir}/EDA_Train", EDA_train)
    # process_files_for_eda_and_features(pre_processed_data_dir, f"{eda_output_dir}/EDA_Test", EDA_test)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Feature Selection
    # feature_selection = FeatureSelection(eda_output_dir, feature_output_dir_1)
    # feature_selection.load_and_process_files()
    # print('Feature Selecting (1) is completed')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Feature Selection
    # feature_selection_2 = FeatureSelection2(eda_output_dir, feature_output_dir_2)
    # feature_selection_2.load_and_process_files()
    # print('Feature Selecting (2) is completed')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Processing   

    processed_output_dir_1.mkdir(parents=True, exist_ok=True)
    processed_output_dir_2.mkdir(parents=True, exist_ok=True)
    # Process each CSV file in the SelectedFeature(1) directory
    for input_file in processing_input_dir_1.glob("*.csv"):  # Make sure it's iterating over files
        print(f"Processing file: {input_file}")
        processing_instance = Processing(str(input_file), str(processed_output_dir_1))
        processing_instance.default_processing()

    # Process each CSV file in the SelectedFeature(2) directory
    # for input_file in processing_input_dir_2.glob("*.csv"):  # Make sure it's iterating over files
    #     print(f"Processing file: {input_file}")
    #     processing_instance = Processing(str(input_file), str(processed_output_dir_2))
    #     processing_instance.default_processing()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    MODELING
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    from models import create_gru_model, train_model, evaluate_model,create_lstm_model,create_rnn_model,create_dnn_model,create_cnn_model
    
    def load_dataset(file_path):
        return pd.read_csv(file_path)

    # Define models to train and test
    models = {
        'GRU': create_gru_model,
        'LSTM': create_lstm_model,
        'RNN': create_rnn_model,
        'DNN': create_dnn_model,
        'CNN': create_cnn_model
    }
    results = pd.DataFrame(columns=['Model', 'Dataset', 'RMSE', 'Score'])

    # Training datasets
    train_data = load_dataset(processed_output_dir_1 / "train_FD001_cleaned_Processed.csv")
    input_shape = (train_data.shape[1] - 1, )  # Adjust based on your data shape and model input

    # Test datasets
    test_datasets = {
        'FD001': load_dataset(processed_output_dir_1 / "test_FD001_cleaned_Processed.csv"),
        'FD002': load_dataset(processed_output_dir_1 / "test_FD002_cleaned_Processed.csv"),
        'FD003': load_dataset(processed_output_dir_1 / "test_FD003_cleaned_Processed.csv"),
        'FD004': load_dataset(processed_output_dir_1 / "test_FD004_cleaned_Processed.csv")
    }

    # Actual RUL values
    actual_rul = {
        'FD001': pd.read_csv(data_root_inputs / "RUL_FD001.txt", header=None, names=['RUL']),
        'FD002': pd.read_csv(data_root_inputs / "RUL_FD002.txt", header=None, names=['RUL']),
        'FD003': pd.read_csv(data_root_inputs / "RUL_FD003.txt", header=None, names=['RUL']),
        'FD004': pd.read_csv(data_root_inputs / "RUL_FD004.txt", header=None, names=['RUL'])
    }

    # Loop through each model
    for model_name, model_func in models.items():
        print(f"Training {model_name} model...")
        model = model_func(input_shape)
        
        # Train the model on FD001
        train_model(model, train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)

        # Evaluate on each FD dataset
        for fd, data in test_datasets.items():
            print(f"Evaluating {model_name} on {fd}...")
            rmse, score = evaluate_model(model, data.iloc[:, :-1].values, actual_rul[fd]['RUL'].values)
            results = results.append({
                'Model': model_name,
                'Dataset': fd,
                'RMSE': rmse,
                'Score': score
            }, ignore_index=True)
            print(f"Finished {model_name} on {fd}: RMSE={rmse}, Score={score}")

    # Save results to CSV
    results.to_csv(data_root / "model_comparison_results.csv", index=False)
    print("All model results saved to 'model_comparison_results.csv'.")


if __name__ == "__main__":
    main()

