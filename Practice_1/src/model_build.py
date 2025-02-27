import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV

class Model:
    def __init__ (self, X, y):
        self.__X = X
        self.__y = y
        self.__svc = None

    def get_X (self):
        return self.__X
    
    def get_y (self):
        return self.__y
    
    def build (self):
        X_train, X_val, y_train, y_val = train_test_split(self.__X, self.__y, test_size=0.2, random_state=42)

        self.__svc = SVC(random_state=42)
        self.__svc.fit(X_train, y_train)

        score_train = self.__svc.score(X_train, y_train)
        score_val = self.__svc.score(X_val, y_val)

        return score_train, score_val, self.__svc
    
    def balancing (self):

        smote = SMOTE(random_state=42)
        self.__X, self.__y = smote.fit_resample(self.__X, )
    
    def evaluate (self, X_test, y_test):
        y_pred = self.__svc.predict(X_test)

        acc = accuracy_score (y_test, y_pred)
        print(f'Accuracy: {acc}\n')

        bas = balanced_accuracy_score(y_test, y_pred)
        print(f'Balanced_accuracy_score: {bas}\n')

        cr = classification_report(y_test, y_pred)
        print(cr)
        print('\n')

    def param_tuning(self):
        X_train, _, y_train, _ = train_test_split(self.__X, self.__y, test_size=0.2, random_state=42)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4]
        }
        
        svm_classifier = SVC()
        
        grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        return best_params, best_model