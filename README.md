# Predicting-Patient-Mortality-Using-ICU-Data

# Predicting Patient Mortality Using ICU Event Data  

## Overview
This project applies **big data analytics and machine learning** to ICU clinical event data to predict **30-day patient mortality after discharge**.  
Using **PySpark** on **Google Colab**, the assignment demonstrates scalable feature engineering and predictive modeling techniques in healthcare analytics.

## Data
Two main datasets were used:
- **events.csv** – Records of patient events, including diagnoses (DIAG), drug prescriptions (DRUG), and lab results (LAB).
- **mortality.csv** – Contains patient IDs of deceased individuals with timestamps.

Each patient’s record was transformed into a structured feature vector using defined **observation (2000 days)** and **prediction (30 days)** windows.


## Methods
### 1. Data Processing (PySpark)
- Computed descriptive statistics (event counts, encounter frequency, record length).
- Filtered and aggregated patient event sequences.
- Constructed sparse feature vectors using **event ID counts**.
- Applied **min-max normalization** and exported data in **SVMLight format**.

### 2. Logistic Regression (SGD)
- Implemented **stochastic gradient descent (SGD)** logistic regression **from scratch** in Python.
- Integrated **L2 regularization** for stability and compared multiple learning rates (η) and regularization parameters (μ).
- Evaluated model performance using **ROC curves** and **AUC metrics**.


## Results
- Built an end-to-end PySpark workflow for transforming raw ICU data into model-ready features.
- Achieved accurate mortality classification using a custom SGD logistic regression implementation.
- Demonstrated how distributed processing and gradient-based optimization can be applied to healthcare-scale data.

## Tech Stack
- **Languages:** Python  
- **Libraries:** PySpark, NumPy, SciPy, scikit-learn  
- **Platform:** Google Colab  
- **Format:** SVMLight for sparse feature representation  

## How to Run
1. Upload the project folder to **Google Drive**.  
2. Open the notebook `BD4H_HW2.ipynb` in **Google Colab**.  
3. Mount Google Drive and update data paths.  
4. Run all cells to reproduce descriptive statistics, feature generation, and model training.  


---

## uthor
**Buddhika Patalee**  
Ph.D. Agricultural Economics | Data Scientist  
[Website](https://www.buddhikapatalee.com) • [LinkedIn](https://www.linkedin.com/in/buddhika-patalee-phd/)  



