# 🌊 **SONAR ROCK vs MINE PREDICTION**
### 🧾 Overview
This project focuses on building a machine learning model to distinguish between rocks and mines (metal cylinders) based on SONAR signal data.
By analyzing reflected SONAR signals, the model predicts whether an underwater object is a rock or a mine, which has applications in defense and underwater navigation.
The project implements a Logistic Regression classifier and achieves high accuracy in classifying the signals.

### 🧰 Tools & Technologies Used
Python
Libraries:
NumPy, Pandas – Data manipulation and analysis
scikit-learn – Model training, testing, and evaluation
LogisticRegression – Classification model
train_test_split, accuracy_score – Model validation
warnings – To handle unnecessary warnings

### 📊 Project Workflow
1. Data Collection
   Dataset: sonar_data.csv
   Each sample contains 60 frequency-based attributes from SONAR readings.
   The target column indicates:
   R → Rock
   M → Mine
2. Data Processing
   Loaded and explored data using Pandas
   Checked for missing values and class distribution
   Converted categorical labels (R/M) into numerical format
3. Splitting the Dataset
   Divided data into training (80%) and testing (20%) sets using:
   `X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)`

🔹 4. Model Training
   Trained a Logistic Regression model on the training data:

    `model = LogisticRegression()`
    `model.fit(X_train, Y_train)`

🔹 5. Model Evaluation
    Calculated accuracy score for both train and test data:

    `train_acc = accuracy_score(Y_train, model.predict(X_train))`
    `test_acc = accuracy_score(Y_test, model.predict(X_test))`



    
