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

## 3.1 Stepped execution information of the code
The following step information is already prompted in the code

  Step1: Pack patient's non-null attributes as a document, which is the set of attribute-value pairs (AVPs).

  Step2: Train the topic model, which named DHTM, to explore the connections between AVPs and therapies.

  Step3: Combine therapies in different categories to form therapy tuples.

  Step4: Calculate the connections between patients and therapy tuples by DHTM.

  Step5: Calculate the connections between patients and therapy tuples by their same key AVPs.

  Step6: Combine these two kinds of connections by a weighted fusion.

  Step7: Calculate the accumulated gain of null attribute for patients.

  Step8: Recommend attributes and evaluate the performance of the model on three metrics.

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

# 6. Symbols in papers

Symbols in papers: Variables in code

$\omega$ ：weight_t1

$\eta$ ： lambda_df

$\chi$：num_PCs

$p'({\Delta | m})$：new_df2

$p(\Delta | m)$ ： new_df1，new_df   

$R_{m,l}$：gain_value