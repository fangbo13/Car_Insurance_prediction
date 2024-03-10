import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn import metrics, tree
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# def print_model_report_and_feature_importance(classifier, data, features, print_feature_importance=True):
#     # Fit the classifier on the data
#     classifier.fit(data[features], data['CLAIM_FLAG'])
        
#     # Predict training set
#     train_predictions = classifier.predict(data[features])
#     train_predprob = classifier.predict_proba(data[features])[:,1]
    
#     # Print model report
#     print("\nModel Report")
#     print("Accuracy : %.4g" % accuracy_score(data['CLAIM_FLAG'].values, train_predictions))
#     print("AUC Score (Train): %f" % roc_auc_score(data['CLAIM_FLAG'], train_predprob))
    
#     # Print Feature Importance
#     if print_feature_importance:
#         feature_importance = pd.Series(classifier.feature_importances_, features).sort_values(ascending=False)
#         feature_importance.plot(kind='bar', title='Feature Importances')
#         plt.ylabel('Feature Importance Score')

        
# def print_model_report_and_coefficients(classifier, data, features):
#     # Fit the classifier on the data
#     classifier.fit(data[features], data['CLAIM_FLAG'])
        
#     # Predict training set
#     train_predictions = classifier.predict(data[features])
#     train_predprob = classifier.predict_proba(data[features])[:,1]
    
#     # Print model report
#     print("\nModel Report")
#     print("Accuracy : %.4g" % accuracy_score(data['CLAIM_FLAG'].values, train_predictions))
#     print("AUC Score (Train): %f" % roc_auc_score(data['CLAIM_FLAG'], train_predprob))
    
#     # Print coefficients
#     print("\nCoefficients:")
#     coef_df = pd.DataFrame({'Feature': features, 'Coefficient': classifier.coef_[0]})
#     print(coef_df)
def print_model_report_and_feature_importance(classifier, data, features, performCV=True, print_feature_importance=True):
    # Fit the classifier on the data
    classifier.fit(data[features], data['CLAIM_FLAG'])
        
    # Predict training set
    train_predictions = classifier.predict(data[features])
    train_predprob = classifier.predict_proba(data[features])[:,1]
    
    # Print model report
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(data['CLAIM_FLAG'].values, train_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(data['CLAIM_FLAG'], train_predprob))
    
    # Perform cross-validation
    if performCV:
        cv_score = cross_val_score(classifier, data[features], data['CLAIM_FLAG'], cv=5, scoring='roc_auc')
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    
    # Print Feature Importance
    if print_feature_importance:
        feature_importance = pd.Series(classifier.feature_importances_, features).sort_values(ascending=False)
        feature_importance.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

        
def print_model_report_and_coefficients(classifier, data, features, performCV=True):
    # Fit the classifier on the data
    classifier.fit(data[features], data['CLAIM_FLAG'])
        
    # Predict training set
    train_predictions = classifier.predict(data[features])
    train_predprob = classifier.predict_proba(data[features])[:,1]
    
    # Print model report
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(data['CLAIM_FLAG'].values, train_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(data['CLAIM_FLAG'], train_predprob))
    
    # Perform cross-validation
    if performCV:
        cv_score = cross_val_score(classifier, data[features], data['CLAIM_FLAG'], cv=5, scoring='roc_auc')
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    
    # Print coefficients
    print("\nCoefficients:")
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': classifier.coef_[0]})
    print(coef_df)
