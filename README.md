# **Decision Tree Classification Project**  

## **Overview**  

This project showcases the use of decision trees for classification tasks. It includes data preprocessing, model training, hyperparameter 
tuning using GridSearchCV, and evaluation using performance metrics such as accuracy, precision, and recall. The dataset undergoes transformations 
to ensure compatibility with machine learning models, making it a comprehensive demonstration of end-to-end classification workflows.

## **Features**  

- **Data Sampling**: Randomized selection of a subset of data for processing and model training.  
- **Label Encoding**: Conversion of categorical variables into numerical form.  
- **Classification Model**: Implementation of a Decision Tree Classifier.
- **Hyperparameter Tuning**: Optimization of model parameters using GridSearchCV.
- **Performance Evaluation**: Assessment with metrics like accuracy, precision, recall, and confusion matrix.

## **Dataset**  

The dataset used for this project includes:  
- **Columns**: Various features with a target variable `y`
- **Size**: Subset sampled to 1,000 records for efficient processing.
- **File**: `PRODIGY_DS_03_Data.csv`  

## **Prerequisites**  

To run this project, ensure you have the following installed:  
- Python 3.7+  
- Libraries:  
  - `pandas`  
  - `scikit-learn`   

Install the required libraries using:  
```  
pip install pandas scikit-learn
```

## **Project Structure**

📂 Decision-Tree-Classification-Project
├── PROGIDY_DS_03.ipynb    # Jupyter Notebook with code and analysis
├── PROGIDY_DS_03_Data.csv # Dataset file
├── README.md              # Project documentation


## **Key Function**

### **Data Pre-processing**
- **Random Sampling**: Selected 1,000 records for analysis.
- **Label Encoding**: Transformed categorical columns into numerical values.

### **Model Training**
used `DecisionTreeClassifier` from `scikit-learn` to fit pre-processed data.

```bash
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### **Hyperparameter Tuning**
Optimized the model using `GridSearchCV` for better accuracy:

```bash
from sklearn.model_selection import GridSearchCV
params = {'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10]}
grid = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
grid.fit(X_train, y_train)
```

### ** Model Evaluation**
Evaluated using:
- Accuracy
- Precision
- Recall
- Confusion matrix

```bash
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
```

## **Results**

- Achieved optimized model performance using GridSearchCV.
- Evaluated and visualized classification metrics for insights.

## **Future Enhancement**
- Add support for feature importance visualization.
- Compare the Decision Tree Classifier with other algorithms like Random Forest or Gradient Boosting.
- Automate hyperparameter tuning using tools like Optuna.

## **Acknowledgement** 
This project was developed during my internship at _Prodigy Infotech_. Special thanks to my mentors and peers for their guidance and support.
