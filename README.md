# Streamlit App for Heart Failure Prediction

![alternative text](img/heart_pic.jpg)


## Project Intro/Objective
The purpose of this project is to create a streamlit app to help predict the presence of heart disease.


### Partners
Ansam Zedan * Silvana Brunner * Nithusya Mohanadasan * Jimena Batschelet

### Methods Used
* Exploratory Data Analysis 
* Supervised Machine Learning
* Deep Learning
* Data Visualization
* Predictive Modeling
* Model Interpretation Model


### Technologies

* Python
* Pandas, jupyter
* plotly
* matplotlib
* seaborn
* Sklearn
* keras
* shap


## Project Description
The dataset for this project was obtained from IEEE Dataport. It is a comprehensive collection that merges five renowned heart disease datasets, namely: Cleveland, Hungarian, Switzerland, Long Beach, VA, and Statlog (Heart) Data Set. The combi-nation of these five distinguished datasets ensures a diverse range of data values, improving the potential generalizability of the model. The dataset encompasses 1,190 instances and includes 11 features: demographic variables (patient age and sex), health metrics (chest pain type, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise angina, oldpeak, ST slope) and the target variable heart disease, indicating the presence or absence of a condition for a patient.

The plan for addressing the research questions is outlined as follows:  

1.	Data Cleaning: Although the dataset is free of missing values, column names contain spaces and require replacing with underscores. Furthermore, outliers in each variable must be checked and treated appropriately. Continuous vari-ables will be scaled using a min-max scalar because it brings all the features into a consistent scale, preventing them from dominating others due to differ-ences in their original ranges. Rescaling features through min-max scaling can lead to better model performance, reducing the chances of the model making incorrect predictions or being influenced more by specific features.

2.	Data Analysis: We will employ Python to apply logistic regression, random for-est, gradient-boosted trees, and neural network models to answer the ques-tions. For this, the data will be split into training and testing sets. The models will then be trained, and the hyperparameters will be finetuned. 

3.	Data Evaluation: To evaluate and compare the models effectively, we will use metrics such as accuracy, precision, recall, and ROC curve. 

4.	Data Visualization: In each of the aforementioned steps, data visualizations (such as histograms and model assessment charts) will be applied to view and verify the implementation.

In conclusion, our structured plan equips us to successfully address our research questions. We are confident that our collective expertise will allow us to overcome any unforeseen technical challenges encountered during the project.


