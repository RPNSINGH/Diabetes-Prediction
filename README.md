# Diabetes-Prediction
Early diabetes detection using ML models . 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug Request](#bug-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [Credits](#credits)


## Demo
[!gui](https://github.com/RPNSINGH/Diabetes-Prediction/blob/main/Diabetes%20Detection/images/gui.png)

## Overview
This is a simple Tkinter app trained with machine learning algorithms. The trained model (`Tkinter_GUI`) takes inputs (*ie Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome*) as an input and predicts the Type II diabetes status.

## Motivation
This disease happens either when the pancreas does not produce enough insulin (type 1 diabetes) or when the produced insulin cannot be used efficiently by the body (type 2 diabetes). In type 1 diabetes, insulin-producing cells in the pancreas are destroyed. On the other hand, type 2 diabetes, usually occurs only after the age of 30 and is therefore often referred to as old-age diabetes. However, WHO reported also that between 24% and 62% of people with type 2 diabetes were undiagnosed and untreated.
Diabetes mellitus (DM) is a form of metabolic disorder whereby the patients suffer high blood sugar levels because their bodies do not respond to, or produce inadequate,  insulin—a hormone that helps to stabilize the blood sugar (glucose) level by directing the cells to take up glucose and inhibit hepatic glucose production.
IoT devices can help doctors make better decisions because the data collected by these devices are highly accurate. We can also take advantage of this advancement to collect health data in order to use it to predict future diabetes based on machine learning (ML)
algorithms.
### Risk Factors of Type II Diabetes:
 
1.	Obesity
2.	Sedentary Lifestyle
3.	Ageing
4.	Sex and Gender
5.	Hypertension
6.	Smoking
7.	Alcohol

## Technical Aspect
This project is divided into four part:
1. Data analysis and data preprocessing :
   - Using pandas to open and manipulate CSV file in jupyter notebook.
   - Visulization using :
     - Seaborn 
     - Matplotlib
   - Label Encoading 
   - Standardization
2. Outliers detection and removal using : 
    - Percentile method
    - Z score method
    - IOR method
3. Model selection and model training:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Classifier
  - Naïve Bayes Classifier
  - K-Nearest Neighbour(KNN)

4. GUI (Tkinter)

## Installation
The Code is written in Python 3.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries  
## DataSets
-  [PimaIndiansdiabetes ](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
-  [Hospital Frankfurt Germany diabetes dataset](https://www.kaggle.com/datasets/johndasilva/diabetes)

## Run
[Python](https://www.python.org/downloads/)<br>
[Pandas](https://pandas.pydata.org/)<br>
[Numpy](https://numpy.org/install/)<br>
[Matplotlib](https://matplotlib.org/stable/users/installing.html)<br>
[Seaborn](https://seaborn.pydata.org/installing.html)<br>
[Sci-kit Learn](https://scikit-learn.org/stable/install.html)<br>
[Jupyter notebook](https://jupyter.org/install)<br>

__Attention__: Please perform the steps given in these tutorials at your own risk. Please don't mess up with the System Variables. It can potentially damage your PC. __You should know what you're doing__. 
- https://www.tenforums.com/tutorials/121855-edit-user-system-environment-variables-windows.html
- https://www.onmsft.com/how-to/how-to-set-an-environment-variable-in-windows-10
## Directory Tree 
```
├── code 
│   ├── final.py
│   ├── data_analysis.ipynb
│   ├── models_training.ipynb
│   ├── outliers_detectiona_removal.ipynb
│   └── TKINTER_GUI.ipynb
├── datasets
│   ├──PimaIndiansdiabetes.csv
│   ├──FrankfurtGermanyDiabetes.csv
│   ├──FinalDiabetesDataset(Corr).csv
│   ├──FinalDiabetesDataset(Avg)
├── images

```

## To Do
1. Convert the app to Django plateform.
2. Add a better vizualization chart to display the predictions.

## Bug Request
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/issues/new). Please include sample queries and their corresponding results.

## Technologies Used
![python](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/blob/main/Bank_loan_prediction/images/python.png)
![pandas](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/blob/main/Bank_loan_prediction/images/pandas.png)
![numpy](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/blob/main/Bank_loan_prediction/images/numpy.png)
![matplot](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/blob/main/Bank_loan_prediction/images/matplot.jpg)
![seaborn](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/blob/main/Bank_loan_prediction/images/seaborn.png)
![sci](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/blob/main/Bank_loan_prediction/images/sci.png)
![jupyter](https://github.com/RPNSINGH/Bank_Loan_Prediction_System/blob/main/Bank_loan_prediction/images/jupyter.png)

## Team
[![RPN](https://github.com/RPNSINGH/RPNSINGH/blob/main/RPN.jpg)] |
-|
[Ratanjeet Pratap Narayan Singh]|)

## Credits
- [Kaggle Datasets](https://www.kaggle.com/datasets) - This project wouldn't have been possible without this tool. It saved my enormous amount of time while collecting the data. A huge regards to [Dr Indu Chawla](https://www.jiit.ac.in/dr-indu-chawla)

