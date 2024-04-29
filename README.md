# Movie Genre Classification

## Introduction:

This repository contains code for a machine learning model aimed at predicting the genre of a movie based on its plot summary or textual information. The model utilizes techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings in combination with classifiers like Naive Bayes, Logistic Regression, and Support Vector Machines.

## Dataset:

The IMDb genre classification dataset, sourced from Kaggle, serves as the foundation for this project. This dataset comprises movie titles along with their corresponding genres, providing a comprehensive collection of films covering various genres such as drama, comedy, horror, thriller, and more.

## Files:

1. `movie_genre_classification.ipynb`: Jupyter Notebook containing the code for data loading, preprocessing, model training, evaluation, and hyperparameter tuning.
2. `train_data.txt`: Text file containing training data with columns for title, genre, and description.
3. `test_data.txt`: Text file containing test data with columns for title and description.

## Results:

Below are the key results obtained from the model evaluation on the test set:

| Metric    | Value    |
|-----------|----------|
| Accuracy  | 42.1%    |
| Precision | 0.4-0.5  |
| Recall    | 0.4-0.5  |
| F1-score  | 0.4-0.5  |

The model demonstrates reasonable performance given the complexity of the task and the imbalance in class distribution. Further improvements could be explored through the use of more advanced models, additional feature engineering, and hyperparameter optimization.

## Conclusion:

This project showcases a Linear SVC model solution for movie genre classification, offering insights into the diverse categorization of films. By leveraging natural language processing techniques and classification algorithms, the model provides a foundation for genre prediction based on textual information.

## Acknowledgements

- The SMS Spam Collection dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).
- Libraries used: NumPy, pandas, scikit-learn, spaCy, seaborn, Matplotlib, imbalanced-learn.

