

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mpl_toolkits.mplot3d import Axes3D
import warnings



# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================
# 1. DATA PREPARATION
# ============================================
print("=" * 70)
print("1. DATA PREPARATION")
print("=" * 70)

# 1.1. Read Data
df = pd.read_csv(r"D:\Dataset\House price\Housing.csv")
print(f"\n1.1 Dataset Shape: {df.shape}")

# 1.2 Random Sample
print("\n1.2 Random Sample:")
print(df.sample(3, random_state=42))

# 1.3 Data Info
print("\n1.3 Column Info:")
print(df.dtypes.value_counts().to_string())

# 1.4 Check Balance
print(f"\n1.4 Price Range: {df['price'].min():,.0f} - {df['price'].max():,.0f}")

# 1.5 Data Cleaning
df = df.drop_duplicates()
df.columns = df.columns.str.lower().str.strip()

# Convert binary columns to 0/1
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Standardize furnishing status
furnish_map = {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
df['furnishingstatus'] = df['furnishingstatus'].str.lower().map(furnish_map)

print(f"\n1.6 After cleaning: {df.shape[0]} rows, {df.shape[1]} cols")
print(f"Missing values: {df.isnull().sum().sum()}")

# 1.7 Outlier Flagging (IQR)
numerical_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
outlier_count = 0
for col in numerical_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    outlier_count += outliers
print(f"Total outliers detected: {outlier_count}")

# ============================================
# 1.8 EDA - Univariate Analysis
# ============================================
print("\n" + "=" * 70)
print("1.8 EDA - UNIVARIATE ANALYSIS")
print("=" * 70)

# Distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, col in enumerate(numerical_cols):
    axes[idx // 3, idx % 3].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx // 3, idx % 3].set_title(f'{col} Distribution')
plt.tight_layout()
plt.savefig('1_distributions.png')
plt.show()

# Skewness & Kurtosis
skew_kurt = pd.DataFrame({
    'Skewness': df[numerical_cols].skew(),
    'Kurtosis': df[numerical_cols].kurtosis()
})
print("\nSkewness & Kurtosis:\n", skew_kurt)

# Log transform for skewed features
df_log = df.copy()
for col in ['price', 'area']:
    df_log[f'{col}_log'] = np.log1p(df[col])
    print(f"{col} skewness: {df[col].skew():.2f} → {df_log[f'{col}_log'].skew():.2f}")

# ============================================
# 1.9 EDA - Bivariate Analysis
# ============================================
print("\n" + "=" * 70)
print("1.9 EDA - BIVARIATE ANALYSIS")
print("=" * 70)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('2_correlation.png')
plt.show()

# Boxplots (Categorical vs Price)
cat_cols = ['mainroad', 'airconditioning', 'prefarea', 'furnishingstatus']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(cat_cols):
    df.boxplot(column='price', by=col, ax=axes[idx // 2, idx % 2])
    axes[idx // 2, idx % 2].set_title(f'Price by {col}')
plt.tight_layout()
plt.savefig('3_boxplots.png')
plt.show()

# Barplots (Mean Price by Category)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(cat_cols):
    df.groupby(col)['price'].mean().plot(kind='bar', ax=axes[idx // 2, idx % 2], color='coral')
    axes[idx // 2, idx % 2].set_title(f'Mean Price by {col}')
plt.tight_layout()
plt.savefig('4_barplots.png')
plt.show()

# ============================================
# 1.10 EDA - Multivariate Analysis
# ============================================
print("\n" + "=" * 70)
print("1.10 EDA - MULTIVARIATE ANALYSIS")
print("=" * 70)

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['area'], df['bedrooms'], df['price'], c=df['price'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Area');
ax.set_ylabel('Bedrooms');
ax.set_zlabel('Price')
plt.title('3D Scatter: Area vs Bedrooms vs Price')
plt.savefig('5_3d_scatter.png')
plt.show()

# VIF (Multicollinearity)
X_vif = df[numerical_cols].dropna()
vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\nVariance Inflation Factor:\n", vif_data)

# PCA
scaler_pca = StandardScaler()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaler_pca.fit_transform(df[numerical_cols]))
df['pca1'], df['pca2'] = pca_result[:, 0], pca_result[:, 1]
print(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(10, 6))
plt.scatter(df['pca1'], df['pca2'], c=df['price'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Price')
plt.title('PCA Visualization')
plt.savefig('6_pca.png')
plt.show()

# ============================================
# 1.11 Data Preprocessing
# ============================================
print("\n" + "=" * 70)
print("1.11 DATA PREPROCESSING")
print("=" * 70)

# Prepare features
feature_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'] + binary_cols + ['furnishingstatus']
X = df[feature_cols].copy()
y = df['price']

# Cap outliers
for col in numerical_cols[1:]:  # exclude price
    Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    X[col] = X[col].clip(lower, upper)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================
# 2. MANDATORY AI TASKS
# ============================================
print("\n" + "=" * 70)
print("2. MANDATORY AI TASKS")
print("=" * 70)

# 2.1 CLUSTERING (Unsupervised)
print("\n2.1 CLUSTERING")
X_cluster = StandardScaler().fit_transform(X)

# Find optimal k
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sil_scores.append(silhouette_score(X_cluster, kmeans.fit_predict(X_cluster)))

optimal_k = range(2, 11)[np.argmax(sil_scores)]
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster)

# 2.2 Clustering Evaluation
sil_score = silhouette_score(X_cluster, cluster_labels)
dbi_score = davies_bouldin_score(X_cluster, cluster_labels)

print(f"Optimal clusters: {optimal_k}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {dbi_score:.4f}")

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['pca1'], df['pca2'], c=cluster_labels, cmap='tab10', alpha=0.6)
plt.colorbar(label='Cluster')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.savefig('7_clustering.png')
plt.show()

# 2.3 CLASSIFICATION (Supervised)
print("\n2.3 CLASSIFICATION")
# Create binary target
y_class = (y > y.median()).astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_c_scaled, y_train_c)
y_pred_c = rf_clf.predict(X_test_c_scaled)
y_pred_proba = rf_clf.predict_proba(X_test_c_scaled)[:, 1]

# 2.4 Classification Metrics
accuracy = accuracy_score(y_test_c, y_pred_c)
f1 = f1_score(y_test_c, y_pred_c)
roc_auc = roc_auc_score(y_test_c, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=['Low Price', 'High Price']))

# Feature importance
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': rf_clf.feature_importances_}).sort_values('importance',
                                                                                                          ascending=False)
print("\nTop 5 Features:\n", feat_imp.head())

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(10), x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - Classification')
plt.savefig('8_feature_importance.png')
plt.show()

# ============================================
# OPTION B: REGRESSION
# ============================================
print("\n" + "=" * 70)
print("OPTION B: REGRESSION")
print("=" * 70)

# Prepare regression data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

# Models
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
for name, model in reg_models.items():
    model.fit(X_train_r_scaled, y_train_r)
    y_pred = model.predict(X_test_r_scaled)

    mae = mean_absolute_error(y_test_r, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    r2 = r2_score(y_test_r, y_pred)

    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2})
    print(f"\n{name}: MAE={mae:,.0f}, RMSE={rmse:,.0f}, R²={r2:.4f}")

# Results summary
results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print("\n" + "-" * 50)
print("REGRESSION RESULTS SUMMARY")
print(results_df.to_string(index=False))

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['MAE', 'RMSE', 'R²']
colors = ['skyblue', 'lightcoral', 'lightgreen']
for idx, (metric, color) in enumerate(zip(metrics, colors)):
    axes[idx].bar(results_df['Model'], results_df[metric], color=color, edgecolor='black')
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('9_regression_comparison.png')
plt.show()

# Best model predictions
best_model = reg_models[results_df.iloc[0]['Model']]
best_model.fit(X_train_r_scaled, y_train_r)
y_pred_best = best_model.predict(X_test_r_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test_r, y_pred_best, alpha=0.5, edgecolors='black')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.xlabel('Actual Price');
plt.ylabel('Predicted Price')
plt.title(f'Best Model: {results_df.iloc[0]["Model"]}\nR² = {results_df.iloc[0]["R²"]:.4f}')
plt.savefig('10_best_model_predictions.png')
plt.show()

# Residual analysis
residuals = y_test_r - y_pred_best
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(y_pred_best, residuals, alpha=0.5);
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted');
axes[0].set_ylabel('Residuals');
axes[0].set_title('Residual Plot')
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7);
axes[1].set_title('Residual Distribution')
stats.probplot(residuals, dist="norm", plot=axes[2]);
axes[2].set_title('Q-Q Plot')
plt.tight_layout()
plt.savefig('11_residual_analysis.png')
plt.show()

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
 DATA: {len(df)} samples, {len(feature_cols)} features
 CLUSTERING: k={optimal_k}, Silhouette={sil_score:.3f}, DBI={dbi_score:.3f}
 CLASSIFICATION: Acc={accuracy:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}
 REGRESSION (Best): {results_df.iloc[0]['Model']} - R²={results_df.iloc[0]['R²']:.3f}
""")
