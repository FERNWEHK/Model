from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score

data = pd.read_csv(r"E:\Model\data.csv")

X = data.iloc[:, 0:6]
y = data.iloc[:, 6]

y_class = np.where(y > 0, "Left", np.where(y < 0, "Right", "Achiral"))
y_encoded = np.array(y_class)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

classifiers = {
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "SVM": SVC(),
    "LR": LogisticRegression(multi_class='multinomial'),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "LightGBM": LGBMClassifier()
}

results = {}
for name, classifier in classifiers.items():
    scores = cross_val_score(classifier, X_train, y_train, cv=10)
    results[name] = scores.mean()
    print(f"{name}: {scores.mean()}")

#
labels, counts = np.unique(y_class, return_counts=True)
sizes = counts / len(y_class) * 100

colors = sns.color_palette("Oranges")

fig, ax = plt.subplots(figsize=(6, 6))
#plt.pie(class_counts, labels=class_counts.index, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 22}, colors=colors)
#ax.set_title('Class Distribution')

plt.axis('equal')
plt.savefig("model_9", dpi=300, bbox_inches='tight')
plt.show()

# Accuracy comparison
plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="Oranges")
plt.xlabel('Models', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("model_31", dpi=300, bbox_inches='tight')
plt.show()

# Find the best model
best_classifier = max(results, key=results.get)
print(f"Best classifier: {best_classifier}")

plt.figure(figsize=(8, 6))
best_classifier = max(results, key=results.get)
classifier = classifiers[best_classifier]
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_mat, annot=True, cmap="Oranges", fmt="d", xticklabels=np.unique(y_encoded), yticklabels=np.unique(y_encoded), annot_kws={"size": 20})
plt.xlabel('Predicted', fontsize=24)
plt.ylabel('Actual', fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("Confusion Matrix", dpi=600)
plt.show()

# Use the best model to predict new data
new_data = pd.read_csv(r"E:\Model\dataset.csv")
X_new = new_data.iloc[:, 0:6]  # Now we are using all six features
new_preds = best_model.predict(X_new)

# Save the predictions to a csv file
datasetpreclass = pd.DataFrame(new_preds, columns=['Predicted_Class'])
datasetpreclass.to_csv("datasetpreclass.csv", index=False)

# After the classifiers are trained, get the feature importance from RandomForest
rf_model = classifiers['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
important_features = indices[:2]

# Now plot the decision boundary using only the two most important features
X_important = X.iloc[:, important_features]
X_train_important, X_test_important, y_train, y_test = train_test_split(X_important, y_class, test_size=0.2, random_state=42)

# Train the best model again with only the two most important features
best_model.fit(X_train_important, y_train)

# Use the best model to predict new data
X_new_important = X_new.iloc[:, important_features]
new_preds_important = best_model.predict(X_new_important)

# Plot decision boundary
h = .02  # step size in the mesh
x_min, x_max = X_new_important.iloc[:, 0].min() - 1, X_new_important.iloc[:, 0].max() + 1
y_min, y_max = X_new_important.iloc[:, 1].min() - 1, X_new_important.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
plt.scatter(X_new_important.iloc[:, 0], X_new_important.iloc[:, 1], c=new_preds_important, edgecolors='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
plt.show()