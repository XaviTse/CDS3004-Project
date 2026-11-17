import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

# ==========================================
# 1. Load Data (Replace with pd.read_csv('hour.csv') for UCI dataset)
# ==========================================
np.random.seed(42)
n_samples = 17379
dates = pd.date_range(start='2011-01-01', periods=n_samples, freq='h')
df = pd.DataFrame({
    'hr': dates.hour,
    'weekday': dates.weekday,
    'mnth': dates.month,
})
df['workingday'] = (df['weekday'] < 5).astype(int)
df['season'] = ((df['mnth'] % 12 + 3) // 3)
df['temp'] = np.clip(0.5 + 0.3 * np.sin((df['mnth'] - 4) * np.pi / 6) + np.random.normal(0, 0.08, n_samples), 0.02, 0.98)
df['hum'] = np.clip(np.random.normal(0.62, 0.19, n_samples), 0.1, 1.0)
df['windspeed'] = np.clip(np.random.exponential(0.19, n_samples), 0.0, 0.85)
df['weathersit'] = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.65, 0.25, 0.09, 0.01])

is_morning = df['hr'].isin([7, 8, 9]) & (df['workingday'] == 1)
is_evening = df['hr'].isin([17, 18, 19]) & (df['workingday'] == 1)
weather_mult = df['weathersit'].map({1: 1.0, 2: 0.85, 3: 0.5, 4: 0.2})
expected_cnt = (50 + 280 * is_morning + 310 * is_evening + 220 * df['temp'] - 90 * df['hum'] - 40 * df['windspeed']) * weather_mult
df['cnt'] = np.maximum(1, np.random.poisson(np.maximum(5, expected_cnt)))

# ==========================================
# 2. Filter to Morning Commute & Engineer Target (`Is_Surge`)
# ==========================================
morning_df = df[df['hr'].isin([7, 8, 9])].copy()

# Define Surge (Class 1) as exceeding the 75th percentile of morning rentals
surge_threshold = morning_df['cnt'].quantile(0.75)
morning_df['Is_Surge'] = (morning_df['cnt'] > surge_threshold).astype(int)

# ==========================================
# 3. Preprocessing Pipeline (Prevent Data Leakage)
# ==========================================
num_features = ['temp', 'hum', 'windspeed']
cat_features = ['hr', 'workingday', 'weekday', 'season', 'weathersit']

X = morning_df[num_features + cat_features]
y = morning_df['Is_Surge']

# Stratified 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Z-Score Standardization for numerical weather features + One-Hot Encoding for categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ]
)

# ==========================================
# 4. Train & Compare CDS3004 Classifiers
# ==========================================
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'SVM (RBF Kernel)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
}

results = []
for name, clf in models.items():
    pipe = Pipeline(steps=[('prep', preprocessor), ('model', clf)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'Model': name,
        'Precision': round(report['1']['precision'], 3),
        'Recall': round(report['1']['recall'], 3),
        'F1-Score': round(report['1']['f1-score'], 3),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob), 3)
    })

print(pd.DataFrame(results).to_string(index=False))