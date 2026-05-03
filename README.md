# ml-classifier-comparison

## about this project 
This project entails a machine learning classification and dimensionality analysis using scikit-learn.
- Part I trains five different classifiers on a chessboard-style dataset, tunes their hyperparameters using cross-validation, and plots the decision boundaries for each.
- Part II explores what happens to nearest-neighbor distances as you increase the number of dimensions

  
## Project Structure
```
hw4/
    PART 1
    classifiers.py                      # All five classifiers, grid search, and decision boundary plots
    input.csv                           # Input dataset for Part I
    output.csv                          # Best training and test accuracies for each method

    PART 2
    curse.py                            # Curse of dimensionality experiment and plot
    ml-classifiers-analysis.pdf         # Plots and written analysis
```

# Part I (Classifier Analysis)
## Classifiers Implemented (each method and their parameters tuned):
- K-Nearest Neighbors (n_neighbors, leaf_size)
- Logistic Regression (C)
- Decision Tree       (max_depth, min_samples_split)
- Random Forest       (max_depth, min_samples_split
- AdaBoost            (n_estimators)

Each classifier is tuned using 5-fold cross-validation and GridSearchCV. The best model from each is then evaluated on the held-out test set. The results are evaluated in the fule `ml-classifiers-analysis.pdf` 

### Requirements
scikit-learn | matplotlib | pandas | numpy

### How to Run
1. Install dependencies:
`pip install scikit-learn matplotlib pandas numpy` 
2. Run the classifiers:
`python classifiers.py` 
3. Run the dimensionality experiment:
`python curse.py` 
4. Output
`output.csv` contains one row per classifier with the best training accuracy and test accuracy:

# Part II (Curse of Dimensionality) 
`curse.py` generates data from a standard Gaussian distribution across dimensions d=1 to d=100, picks a random query point, and tracks the DMAX/DMIN ratio as dimensionality grows. As d increases, the nearest and furthest points become roughly the same distance away, which makes KNN increasingly unreliable.
- The report also covers research on why high dimensionality can sometimes work in your favor, including cases in text classification and kernel methods where more dimensions actually help separability.
