# Machine Learning in NLP - Assignment 1

Welcome to my solution for Assignment 1 in the Machine Learning in NLP course. This assignment focuses on language identification using different machine learning techniques, including linear classifiers and neural networks. Below, I explain the tasks, the flow of the solution, and what I did step-by-step.

## Overview

The assignment is divided into two main parts:
1. **Language Identification with Linear Classification**
2. **Building a Neural Network for Language Identification**

Each part involves specific tasks that help us understand and implement machine learning models for language classification.

## Part 1: Language Identification with Linear Classification

### Task 1: Loading and Preprocessing Data

In this task, we start by loading the dataset, which consists of text segments in different languages. The goal is to classify these text segments correctly.

- **Data Loading**: I loaded the training and test data from the provided text files (`x_train.txt`, `y_train.txt`, `x_test.txt`, `y_test.txt`).
- **Data Preprocessing**: I created a pipeline using `scikit-learn` to preprocess the data. This involved converting the text data into numerical features using techniques like tokenization and vectorization.

### Task 2: Training a Logistic Regression Classifier

Next, I trained a logistic regression classifier to identify the languages.

- **Model Training**: I used `LogisticRegression` from `scikit-learn` and experimented with different hyperparameters using `GridSearchCV`.
- **Hyperparameter Tuning**: I tuned parameters like penalty and solver to find the best model.
- **Error Analysis**: I evaluated the model using a confusion matrix and identified areas where the model performed well and where it needed improvement.

### Task 3: Feature Importance and Model Explanation

To understand the model better, I analyzed the feature importance.

- **Feature Importance**: Using the `ELI5` library, I generated a feature importance table for the top features in languages like English, Swedish, Norwegian, and Japanese.
- **Model Explanation**: This helped in explaining what the model learned and which features were most significant in making predictions.

### Task 4: Ablation Study

Finally, I conducted an ablation study to see how reducing the number of characters in the text segments affects the model's performance.

- **Ablation Study**: I re-fitted the model several times with reduced character lengths (all characters, 500 characters, 100 characters) and observed the changes in performance.

## Part 2: Building a Neural Network for Language Identification

### Task 1: Initial Neural Network Setup

I started by setting up a simple neural network using `skorch`, a `scikit-learn` compatible neural network library.

- **Neural Network Setup**: I built a neural network and trained it on the same dataset used in Part 1.
- **Initial Results**: The initial model was evaluated to see its performance compared to the logistic regression model.

### Task 2: Improving the Neural Network

To improve the neural network's performance, I experimented with various hyperparameters.

- **Hyperparameter Tuning**: I played around with different layer sizes, activation functions, solvers, and early stopping techniques.
- **Performance Improvement**: My goal was to achieve at least 80% accuracy. I reported the best combination of hyperparameters that resulted in the highest performance.

## Conclusion

This assignment helped me understand the process of building and evaluating machine learning models for language identification. I learned how to preprocess data, train different models, tune hyperparameters, and interpret model results. The skills gained from this assignment are essential for working on more complex NLP tasks.

## Files in This Repository

   - `x_train.txt`: Training data features.
  - `y_train.txt`: Training data labels.
  - `x_test.txt`: Testing data features.
  - `y_test.txt`: Testing data labels.

## Required Libraries

To run the notebooks, you need to have the following libraries installed:

- `scikit-learn`
- `skorch`
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `eli5`

You can install them using `pip`:

```bash
pip install scikit-learn skorch torch numpy pandas matplotlib eli5

Feel free to explore the notebooks and see the code in action. If you have any questions or suggestions, please let me know!

Happy Learning!

