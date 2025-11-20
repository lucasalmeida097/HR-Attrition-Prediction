from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

def get_base_pipelines(preprocessor):
    pipelines = {}
    pipelines['logistic'] = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))])
    pipelines['random_forest'] = Pipeline([('pre', preprocessor),('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1))])
    pipelines['gb'] = Pipeline([('pre', preprocessor), ('model', GradientBoostingClassifier(random_state=42))])
    pipelines['svc'] = Pipeline([('pre', preprocessor), ('model', SVC(probability=True))])
    
    return pipelines

def calibrate_model(pipeline, X_train, y_train, method='isotonic', cv=3):
    calibrated = CalibratedClassifierCV(base_estimator=pipeline, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    
    return calibrated

