# Arvato-Customer-Segmentation
Customer Segmentation repository for Arvato Financial Services.


## Requirements

Package requirements are described in `requirements.txt`. Additional requirement is the access to AWS cloud for Sage Maker used for hyperparameter tuning. 

## Project structure

### Project proposal
Project proposal could be found in [Proposal/proposal.pdf](Proposal/proposal.pdf).

### Project report
Project report could be found in [Report/ArvatoReport.pdf](Report/ArvatoReport.pdf).

### Analysis

Analysis is split into multiple folders corresponding to various steps, also described in the project report.

1. Folder `EDA`:
    Contains the Exploratory Data Analysis Steps along with concating and imputation steps of data processing. Notebooks names are starting with an intiger indicating the order in which they should be executed/read. 
    
    - `01_EDA_AZDIAS.ipynb`
    - `02_EDA_CUSTOMERS.ipynb`
    - `03_Concatenate_and_Impute_Azdias_Customers.ipynb`
    - `04_MAILOUT.ipynb`

2. Folder `Unsupervised`
    Contains the major part of this project, namely the customer segmentation projects that identifies and characterizes the market segments. Explores its meaning, and charactierizes its importance. 
    - `Unsupervised_Customer_Segmentation.ipynb`
    

3. Folder `Supervised`
    Contains two notebooks for two **types** of models tested for prediction of successful response of a customer after being exposed to a marketing campaign. 
    - `01_LinearLerner.ipynb`: a baseline linear model
    - `02_XGBoost.ipynb`: tree-boosted model with hyperparameter tuning


4. Folder `submissions`
contains the final .csv files for the submissions to the Kaggle competition.



