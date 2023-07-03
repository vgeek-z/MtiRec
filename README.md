# 1. Virtual Environment
Input the following code to create the virtual environment MtiRec

    conda env create -f environment.yaml

# 2. Project Directory


```
│ DHTMRec.py            # Training + prediction

│ Environment.YAMl      # Environment

│  README.md            

├─data                  # Folder : store raw data

│ *1

│

├─model

│ │ assessment_criteria.py    # Evaluation metrics

│ │ data_process.py           # Data processing

│ │ DHtm.py                   # Topic model : DHTM

│ │ mti_rec.py                # Some tools 

│ └─__pycache__               # Store '.pyc' files

│

└─result_data                 # Save results related to model training and prediction
  dhtm_model.pkl              # A trained model dhtm
  terms.xlsx                  # AVPs (*2)
  test_df.xlsx                # Test set  (*2)
  train_df.xlsx               # Training set (*2)
  Accuracy_lambda50_weight20_bench5.xlsx 
```


*1: data can be downloaded from the BCDB's official website: http://bcdb.mdt.team:8080

*2: Only tiny data as an example


# 3. Train and predict
Input the following code to run the project

    python DHTMRec.py

# 4. Expected Running Results
## 4.1 Progress Prompt Messages
Prompts for the completion of dataset processing, Gibbs sampling progress, tuple information, etc.
## 4.2 Model Saving
The DHTM model is saved in the "result_data" folder as "dhtm_model.pkl".
## 4.3 Prediction Results
Save the evaluation metrics Accuracy and MRR of the test set in the "result_data" folder. The file name should correspond to the parameter settings for the evaluation metrics results, such as "Accuracy_lambda50_weight20_bench5.xlsx", with the respective parameters being 0.5, 0.2, and 5.

# 5. Other Information
1. Due to privacy protection and other reasons, medical data is not uploaded. You can apply for downloading from the BCDB official website (http://bcdb.mdt.team:8080).

2. DHTMRec.py provides a multi-threading method for model performance evaluation to determine the optimal parameters for custom datasets.