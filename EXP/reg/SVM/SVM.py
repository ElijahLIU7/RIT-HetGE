import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
import numpy as np
from sklearn.manifold import TSNE


df = pd.read_csv('real&random_AAC.csv')

df_class_0 = df[df['label'] == 0]
df_class_1 = df[df['label'] == 1]

# X_train_class_0, X_test_class_0 = train_test_split(df_class_0, test_size=0.2, random_state=42)
# X_train_class_1, X_test_class_1 = train_test_split(df_class_1, test_size=0.2, random_state=42)
X_train_class_0, X_test_class_0 = train_test_split(df_class_0, test_size=0.2)
X_train_class_1, X_test_class_1 = train_test_split(df_class_1, test_size=0.2)

X_train = pd.concat([X_train_class_0, X_train_class_1])
X_test = pd.concat([X_test_class_0, X_test_class_1])

X_train = X_train.iloc[:, 2:]
X_test = X_test.iloc[:, 2:]
y_train = df.loc[X_train.index, 'label']
y_test = df.loc[X_test.index, 'label']

svm = SVC()

param_grid = {
    'C': [0.1, 0.2, 0.5, 1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

# kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf = KFold(n_splits=10, shuffle=True)

grid_search = GridSearchCV(svm, param_grid, cv=kf)
grid_search.fit(X_train, y_train)

cv_results = grid_search.cv_results_

# 将cv_results转换为DataFrame
cv_results_df = pd.DataFrame(cv_results)

# 将DataFrame写入CSV文件
cv_results_df.to_csv('cv_results.csv', index=False)

cv_results_data = []
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    cv_results_data.append({'Mean Score': mean_score, 'Parameters': params})

best_params = grid_search.best_params_
best_score = grid_search.best_score_

y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


fieldnames = list(cv_results_data[0].keys()) + ['Best Parameters', 'Best Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
results = cv_results_data + [{'Best Parameters': best_params,
                              'Best Score': best_score,
                              'Precision': precision,
                              'Recall': recall,
                              'F1 Score': f1,
                              'Accuracy': accuracy}]
with open('grid_search_results.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("Best parameters: ", best_params)
print("Best score: ", best_score)
print("Test accuracy: ", accuracy)
print("Test precision: ", precision)
print("Test recall: ", recall)
print("Test F1 score: ", f1)


# 计算ROC曲线的假正率和真正率
y_scores = grid_search.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# 绘制ROC曲线
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# 计算混淆矩阵
y_pred = grid_search.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2)
X_train_tsne = tsne.fit_transform(X_train)

# 训练 SVM 模型
svm = SVC(kernel='linear')
svm.fit(X_train_tsne, y_train)

# 绘制数据点
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap=plt.cm.Paired)

# 绘制超平面
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格点来绘制超平面
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# 绘制超平面和边界
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('SVM Hyperplane (t-SNE)')
plt.show()

