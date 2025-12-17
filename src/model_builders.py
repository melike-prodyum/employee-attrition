"""
Model Builder Functions
Decision Tree ve Random Forest modellerini oluşturan fonksiyonlar
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_decision_tree():
    """
    Decision Tree modelini oluşturur ve döndürür.
    
    Returns:
        DecisionTreeClassifier: Yapılandırılmış Decision Tree modeli
    """
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=50,
        criterion='gini',
        random_state=42,
        class_weight='balanced'
    )
    return dt_model


def build_random_forest():
    """
    Random Forest modelini oluşturur ve döndürür.
    
    Returns:
        RandomForestClassifier: Yapılandırılmış Random Forest modeli
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=200,
        min_samples_leaf=100,
        criterion='gini',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        max_features='sqrt'
    )
    return rf_model


def get_decision_tree_params():
    """Decision Tree parametrelerini döndürür."""
    return {
        'max_depth': 5,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'criterion': 'gini',
        'random_state': 42,
        'class_weight': 'balanced'
    }


def get_random_forest_params():
    """Random Forest parametrelerini döndürür."""
    return {
        'n_estimators': 100,
        'max_depth': 4,
        'min_samples_split': 200,
        'min_samples_leaf': 100,
        'criterion': 'gini',
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'max_features': 'sqrt'
    }
