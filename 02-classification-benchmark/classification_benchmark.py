from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.decomposition import PCA
np.random.seed(42)

#  read data
DATA_DIR = Path(__file__).resolve().parent / "data"
dataframe = pd.read_csv(DATA_DIR / "breast-cancer.csv") # pandas
X = dataframe.iloc[:, 2:].values # get the columns from 2 to end as features
y = dataframe.iloc[:, 1].values # get the second column as the target
y = np.where(y == 'M', 1, 0) # convert 'M' to 1 and 'B' to 0 for binary classification

# split data into train and test sets (70% train, 30% test) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
) # 42 is a common choice for random_state 

# evaluation function to calculate accuracy, precision, recall, and F1 score
# args: y_true (true labels), y_pred (predicted labels)
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("-"*30)
    print(f"Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%") 
    print("-"*30)

    return [round(acc*100,2), round(prec*100,2), round(rec*100,2), round(f1*100,2)]

# 1(a) missing value imputation comparison 缺失值填充对比
print("="*30)
print("missing value imputation comparison (mean vs median)")
print("="*30)

imp_mean = SimpleImputer(missing_values=0, strategy='mean')
imp_median = SimpleImputer(missing_values=0, strategy='median')

X_train_mean = imp_mean.fit_transform(X_train)
X_test_mean = imp_mean.transform(X_test) # warning: should use the same imputer fitted on train data to transform test data, otherwise it will cause data leakage and overfitting 注意：测试集必须使用训练集的统计量进行填充，不能重新计算，否则会泄漏信息
X_train_med = imp_median.fit_transform(X_train)
X_test_med = imp_median.transform(X_test)

dt = DecisionTreeClassifier(max_depth=6, random_state=42)

pred_mean = dt.fit(X_train_mean, y_train).predict(X_test_mean)
print("mean imputation ->", evaluate(y_test, dt.predict(X_test_mean)))

pred_median = dt.fit(X_train_med, y_train).predict(X_test_med)
print("median imputation ->", evaluate(y_test, dt.predict(X_test_med)))
 
print("Are predictions equal?", np.array_equal(pred_mean, pred_median))
  
X_train_clean = X_train_med
X_test_clean = X_test_med
   
# 1(b) normalization impact on KNN & Decision Tree  标准化对 KNN 和决策树的影响
print("="*30)
print("normalization impact on KNN & Decision Tree")
print("="*30)
 
dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

dt_model.fit(X_train_clean, y_train)
knn_model.fit(X_train_clean, y_train)

# without normalization
print(" 【Without normalization】")
print("Decision tree ->", evaluate(y_test, dt_model.predict(X_test_clean)))
print("KNN   ->", evaluate(y_test, knn_model.predict(X_test_clean)))

# StandardScaler 
scaler_std = StandardScaler()

X_train_std = scaler_std.fit_transform(X_train_clean)
X_test_std = scaler_std.transform(X_test_clean)
# fit the models on the standardized data
dt_model.fit(X_train_std, y_train)
knn_model.fit(X_train_std, y_train)

print(" 【Standard】") # 【Standard】标准化
print("Decision tree ->", evaluate(y_test, dt_model.predict(X_test_std)))
print("KNN   ->", evaluate(y_test, knn_model.predict(X_test_std)))

# MinMaxScaler
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train_clean)
X_test_mm = scaler_mm.transform(X_test_clean)

dt_model.fit(X_train_mm, y_train)
knn_model.fit(X_train_mm, y_train)

print("【MinMax】") #  【MinMax】归一化
print("Decision tree ->", evaluate(y_test, dt_model.predict(X_test_mm)))
print("KNN   ->", evaluate(y_test, knn_model.predict(X_test_mm)))

# 2. Hyperparameter tuning 超参数调参
print("="*30)
print(" Hyperparameter tuning")
print("="*30)
results = []

# KNN
for k in [3,9,15,21]:
    pred = KNeighborsClassifier(n_neighbors=k).fit(X_train_std, y_train).predict(X_test_std)
    results.append(["KNN", "n_neighbors", k] + evaluate(y_test, pred))

# Decision Tree
for d in [2,8,14]:
    pred = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train_std, y_train).predict(X_test_std)
    results.append(["Decision Tree", "max_depth", d] + evaluate(y_test, pred))

# AdaBoost
for n in [10,20,30]:
    pred = AdaBoostClassifier(n_estimators=n, random_state=42).fit(X_train_std, y_train).predict(X_test_std)
    results.append(["AdaBoost", "n_estimators", n] + evaluate(y_test, pred))

# Random Forest
for n in [10,30,50,60]:
    pred = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train_std, y_train).predict(X_test_std)
    results.append(["Random Forest", "n_estimators", n] + evaluate(y_test, pred))

# Example of printing results  
table = pd.DataFrame(results, columns=["Classifier", "Hyperparameter", "Value", "Accuracy%", "Precision", "Recall", "F1"])
print(table.to_string(index=False))

# 3(a) Pearson feature selection with threshold 0.6 皮尔逊特征选择（阈值0.6）
print("="*30)
print("3(a) Pearson feature selection with threshold 0.6")
print("="*30) 

# compute Pearson correlation between each feature and the target variable
correlations = np.array([
    np.corrcoef(X_train_std[:, i], y_train)[0, 1] 
    for i in range(X_train_std.shape[1])
])
 
# thresholding: select features with absolute correlation > 0.6
selected_mask = np.abs(correlations) > 0.6

# filter the training and test data to keep only the selected features
X_train_sel = X_train_std[:, selected_mask]
X_test_sel = X_test_std[:, selected_mask]

print("Selected features:", X_train_sel.shape[1])

# train a decision tree classifier on the selected features and evaluate its performance
dt_sel = DecisionTreeClassifier(max_depth=6, random_state=42)

#   X_train_clean
print("Before selection ->")
evaluate(y_test, dt_sel.fit(X_train_std, y_train).predict(X_test_std))

#  X_train_sel
print("After selection ->")
evaluate(y_test, dt_sel.fit(X_train_sel, y_train).predict(X_test_sel))

# 3（b） PCA dimensionality reduction PCA 降维
print("="*30)
print("3(b) PCA dimensionality reduction")
print("="*30)

pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

for d in [2,8]:
    dt_pca = DecisionTreeClassifier(max_depth=d, random_state=42)
    print(f" decision tree depth {d} | without pca ->", evaluate(y_test, dt_pca.fit(X_train_std, y_train).predict(X_test_std)))
    print(f" decision tree depth {d} | with pca ->", evaluate(y_test, dt_pca.fit(X_train_pca, y_train).predict(X_test_pca)))

# 4 PCA vs feature selection (same dimension) PCA vs 特征选择（维度相同）
print("="*30)
print("4 PCA vs feature selection (same dimension)")
print("="*30)
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

dt_sel = DecisionTreeClassifier(max_depth=6, random_state=42)

print("Feature selection result:")
evaluate(y_test, dt_sel.fit(X_train_sel, y_train).predict(X_test_sel))

print("PCA result (same 10 dimensions):")
evaluate(y_test, dt_sel.fit(X_train_pca, y_train).predict(X_test_pca))

#  5. 10-fold cross-validation for KNN and Decision Tree
print("="*30)
print("5. 10-fold cross-validation for KNN and Decision Tree")
print("="*30)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
models = [
    ("KNN k=3", KNeighborsClassifier(3)),
    ("KNN k=9", KNeighborsClassifier(9)),
    ("DT depth=2", DecisionTreeClassifier(max_depth=2, random_state=42)),
    ("DT depth=8", DecisionTreeClassifier(max_depth=8, random_state=42))
]

for name, model in models:
    acc_scores = cross_val_score(model, X_train_std, y_train, cv=kf, scoring="accuracy") * 100
    f1_scores = cross_val_score(model, X_train_std, y_train, cv=kf, scoring="f1") * 100

    print(f"\n{name}")
    print("Accuracy per fold:", np.round(acc_scores, 2))
    print("F1 per fold:      ", np.round(f1_scores, 2)) 
    print(f"Mean Accuracy: {np.mean(acc_scores):.2f}% ± {np.std(acc_scores):.2f}")
    print(f"Mean F1:       {np.mean(f1_scores):.2f}% ± {np.std(f1_scores):.2f}")

print("="*30)
