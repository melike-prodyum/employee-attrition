"""
Evaluation Utility Functions
Model değerlendirme ve metrik hesaplama için ortak fonksiyonlar
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


def calculate_metrics(y_true, y_pred, y_proba, prefix=''):
    """
    Model performans metriklerini hesaplar.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        y_proba: Tahmin olasılıkları
        prefix: Metrik isimleri için prefix (örn: 'Train ', 'Val ')
    
    Returns:
        dict: Metrikler dictionary
    """
    metrics = {
        f'{prefix}Accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}Precision': precision_score(y_true, y_pred),
        f'{prefix}Recall': recall_score(y_true, y_pred),
        f'{prefix}F1': f1_score(y_true, y_pred),
        f'{prefix}ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    return metrics


def print_metrics(y_true, y_pred, y_proba, label='Set'):
    """
    Metrikleri yazdırır.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        y_proba: Tahmin olasılıkları
        label: Set etiketi (örn: 'Train', 'Validation')
    """
    print(f"\n🔹 {label} Seti:")
    print(f"  • Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  • Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  • Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  • F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  • ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Detaylı sınıflandırma raporunu yazdırır.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        target_names: Sınıf isimleri
    """
    if target_names is None:
        target_names = ['Not Leave (0)', 'Leave (1)']
    
    print("\n📋 Detaylı Sınıflandırma Raporu (Validation):")
    print("-" * 70)
    print(classification_report(y_true, y_pred, target_names=target_names))


def print_confusion_matrix(y_true, y_pred):
    """
    Confusion matrix yazdırır.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
    
    Returns:
        numpy.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\n🔢 Confusion Matrix (Validation):")
    print(cm)
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    return cm


def print_feature_importance(model, feature_names, top_n=10):
    """
    Feature importance yazdırır.
    
    Args:
        model: Eğitilmiş model
        feature_names: Feature isimleri
        top_n: Gösterilecek en önemli feature sayısı
    
    Returns:
        pd.DataFrame: Feature importance DataFrame
    """
    import pandas as pd
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n⭐ En Önemli Özellikler (Top {top_n}):")
    print("-" * 70)
    for idx, row in feature_importance.head(top_n).iterrows():
        print(f"  {row['Feature']:30s} : {row['Importance']:.4f}")
    
    return feature_importance
