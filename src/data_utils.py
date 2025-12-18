"""
Data Utility Functions
Veri yükleme, ön işleme ve encoding işlemleri için ortak fonksiyonlar
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


def load_data(project_root=None):
    """
    Train, test ve sample_submission veri setlerini yükler.
    
    Args:
        project_root: Proje kök dizini. None ise otomatik hesaplanır.
    
    Returns:
        tuple: (train_df, test_df, submission_df)
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_df = pd.read_csv(os.path.join(project_root, 'data', 'aug_train.csv'))
    test_df = pd.read_csv(os.path.join(project_root, 'data', 'aug_test.csv'))
    submission_df = pd.read_csv(os.path.join(project_root, 'data', 'sample_submission.csv'))
    
    return train_df, test_df, submission_df


def print_data_info(train_df, test_df):
    """
    Veri seti hakkında bilgi yazdırır.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
    """
    print(f"✓ Train veri seti boyutu: {train_df.shape}")
    print(f"✓ Test veri seti boyutu: {test_df.shape}")
    print(f"\nSütunlar: {list(train_df.columns)}")
    
    # Target dağılımı
    print(f"\n📊 Target Dağılımı:")
    print(train_df['target'].value_counts())
    print(f"Target oranı: {train_df['target'].value_counts(normalize=True)}")
    
    # Eksik değerler
    print(f"\n📋 Eksik Değerler:")
    missing = train_df.isnull().sum()
    missing_pct = (missing / len(train_df)) * 100
    missing_df = pd.DataFrame({
        'Eksik Sayısı': missing,
        'Yüzde': missing_pct
    }).sort_values('Eksik Sayısı', ascending=False)
    print(missing_df[missing_df['Eksik Sayısı'] > 0])


def prepare_features(train_df, test_df):
    """
    Features ve target'ı ayırır.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
    
    Returns:
        tuple: (X_train, X_test, y, train_ids, test_ids)
    """
    train_ids = train_df['enrollee_id']
    test_ids = test_df['enrollee_id']
    
    y = train_df['target']
    X_train = train_df.drop(['enrollee_id', 'target'], axis=1)
    X_test = test_df.drop(['enrollee_id'], axis=1)
    
    return X_train, X_test, y, train_ids, test_ids


def get_column_types(X):
    """
    Kategorik ve numerik sütunları ayırır.
    
    Args:
        X: DataFrame
    
    Returns:
        tuple: (categorical_cols, numerical_cols)
    """
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return categorical_cols, numerical_cols


def fill_missing_values(X_train, X_test, categorical_cols, numerical_cols, verbose=True):
    """
    Eksik değerleri doldurur.
    
    Args:
        X_train: Training features
        X_test: Test features
        categorical_cols: Kategorik sütun listesi
        numerical_cols: Numerik sütun listesi
        verbose: Bilgi yazdırılsın mı?
    
    Returns:
        tuple: (X_train, X_test) - Doldurulmuş DataFrame'ler
    """
    if verbose:
        print("\n🔧 Eksik değerleri doldurma:")
    
    # Numerik sütunlar için median
    for col in numerical_cols:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
            if verbose:
                print(f"  - {col}: median ile dolduruldu")
    
    # Kategorik sütunlar için mode
    for col in categorical_cols:
        if X_train[col].isnull().sum() > 0:
            mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'Unknown'
            X_train[col].fillna(mode_val, inplace=True)
            X_test[col].fillna(mode_val, inplace=True)
            if verbose:
                print(f"  - {col}: mode ile dolduruldu")
    
    return X_train, X_test


def apply_one_hot_encoding(X_train, X_test, categorical_cols, verbose=True):
    """
    One-Hot Encoding uygular.
    
    Args:
        X_train: Training features
        X_test: Test features
        categorical_cols: Kategorik sütun listesi
        verbose: Bilgi yazdırılsın mı?
    
    Returns:
        tuple: (X_train_encoded, X_test_encoded)
    """
    if verbose:
        print("\n🔧 Kategorik değişkenleri One-Hot Encoding ile encode etme:")
    
    if not categorical_cols:
        return X_train, X_test
    
    # One-Hot Encoding uygula
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
    
    # Train ve test'te aynı sütunların olmasını sağla
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0
    
    extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
    X_test_encoded = X_test_encoded.drop(columns=extra_cols)
    
    # Sütun sırasını aynı yap
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    if verbose:
        print(f"  ✓ One-Hot Encoding tamamlandı - {X_train_encoded.shape[1]} feature")
        if verbose == 'detailed':
            for col in categorical_cols:
                encoded_cols = [c for c in X_train_encoded.columns if c.startswith(f"{col}_")]
                print(f"  - {col}: {len(encoded_cols)} kategoriye dönüştürüldü")
    
    return X_train_encoded, X_test_encoded


def apply_label_encoding(X_train, X_test, categorical_cols, verbose=True):
    """
    Label Encoding uygular.
    
    Args:
        X_train: Training features
        X_test: Test features
        categorical_cols: Kategorik sütun listesi
        verbose: Bilgi yazdırılsın mı?
    
    Returns:
        tuple: (X_train_encoded, X_test_encoded, label_encoders)
    """
    if verbose:
        print("\n🔧 Kategorik değişkenleri Label Encoding ile encode etme:")
    
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Train ve test'i birleştirerek tüm kategorileri öğren
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
        
        if verbose:
            print(f"  - {col}: {len(le.classes_)} kategori")
    
    return X_train, X_test, label_encoders


def create_output_directory(model_name):
    """
    Output klasörünü oluşturur.
    
    Args:
        model_name: Model adı (decision_tree, random_forest, compare_models)
    
    Returns:
        str: Output klasörünün yolu
    """
    output_dir = f'../outputs/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_submission_file(model, X_test, submission_template, output_filename):
    """
    Test verisi için submission dosyası oluşturur.
    
    Args:
        model: Eğitilmiş model
        X_test: Test features
        submission_template: Sample submission DataFrame veya yolu
        output_filename: Çıktı dosya adı (örn: 'submission_decision_tree.csv')
    
    Returns:
        tuple: (predictions, submission_df) - Tahminler ve submission DataFrame
    """
    # Test tahminleri
    test_predictions = model.predict_proba(X_test)[:, 1]
    
    # Submission template'i oku (eğer string ise)
    if isinstance(submission_template, str):
        submission = pd.read_csv(submission_template)
    else:
        submission = submission_template.copy()
    
    # Tahminleri ekle
    submission['target'] = test_predictions
    
    # Klasör oluştur
    os.makedirs('../submissions', exist_ok=True)
    
    # Dosyayı kaydet
    output_path = f'../submissions/{output_filename}'
    submission.to_csv(output_path, index=False)
    
    return test_predictions, submission
