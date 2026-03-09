import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=200,
    evals=[(dtest, "Test")],
    early_stopping_rounds=20,
    verbose_eval=False
)

y_prob = model.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

cm = confusion_matrix(y_test, y_pred)
im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0, 0].figure.colorbar(im, ax=axes[0, 0])
axes[0, 0].set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Benign', 'Malignant'],
               yticklabels=['Benign', 'Malignant'],
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
axes[0, 0].grid(False)

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(y_prob[y_test == 0], bins=30, alpha=0.7, label='Benign', color='green', edgecolor='black')
axes[1, 0].hist(y_prob[y_test == 1], bins=30, alpha=0.7, label='Malignant', color='red', edgecolor='black')
axes[1, 0].axvline(x=0.5, color='blue', linestyle='--', lw=2, label='Threshold (0.5)')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Prediction Probability Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

feature_names = load_breast_cancer().feature_names
importance = model.get_score(importance_type='weight')
importance_dict = {feature_names[int(k[1:])]: v for k, v in importance.items()}
sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15])

features = list(sorted_importance.keys())
scores = list(sorted_importance.values())
axes[1, 1].barh(features, scores)
axes[1, 1].set_xlabel('Importance Score')
axes[1, 1].set_title('Top 15 Feature Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('xgboost_classification_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'xgboost_classification_results.png'")
plt.show()
