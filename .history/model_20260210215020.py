"""Model training and evaluation module"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config import MODEL_RANDOM_STATE, MODEL_MAX_ITER, TEST_SET_SIZE


class PovertyModel:
    """Logistic Regression model for poverty classification"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.features = None
        
    def train(self, df, features):
        """Train the logistic regression model"""
        self.features = features
        
        # Prepare data
        X = df[features].values
        y = df['high_poverty'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SET_SIZE, random_state=MODEL_RANDOM_STATE
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(random_state=MODEL_RANDOM_STATE, max_iter=MODEL_MAX_ITER)
        self.model.fit(X_train_scaled, y_train)
        
        # Store test data and predictions
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test_scaled)
        self.y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
    def get_accuracy(self):
        """Get model accuracy on test set"""
        return self.model.score(self.X_test, self.y_test)
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        return confusion_matrix(self.y_test, self.y_pred)
    
    def get_roc_curve(self):
        """Get ROC curve data"""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    def predict(self, input_array):
        """Make prediction for new data"""
        input_scaled = self.scaler.transform(input_array)
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]
        return prediction, probability
    
    def get_coefficients(self):
        """Get feature coefficients"""
        return self.model.coef_[0]
