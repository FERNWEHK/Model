import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from catboost import CatBoostRegressor
import lightgbm as lgb

import matplotlib.ticker as ticker

#
data = pd.read_csv(r"E:\Model\data.csv")

#
X = data.iloc[:, 0:6]
y = data.iloc[:, 6]

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#
models = [
    ("DT", DecisionTreeRegressor(random_state=42)),
    ("RF", RandomForestRegressor(random_state=42)),
    ("KNN", KNeighborsRegressor()),
    ("XGBoost", XGBRegressor(random_state=42)),
    ("CatBoost", CatBoostRegressor(random_state=42)),
    ("LightGBM", lgb.LGBMRegressor(random_state=42)),
]

#
results = []
models_dict = {}
for name, model in models:
    model.fit(X_train, y_train)
    models_dict[name] = model
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
    mse_scores = np.sqrt(-scores)
    r2_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="r2")
    results.append((name, np.mean(mse_scores), np.mean(r2_scores)))

#
results_df = pd.DataFrame(results, columns=["Model", "MSE", "R²"])
#
best_model_name = max(results, key=lambda x: x[2])[0]
best_model = models_dict[best_model_name]

print("Best Model:", best_model_name)
print("RMSE:", results_df.loc[results_df['Model'] == best_model_name, 'MSE'].values[0])
print("R²:", results_df.loc[results_df['Model'] == best_model_name, 'R²'].values[0])

results = []
models_dict = {}

fig, axs = plt.subplots(2, 3, figsize=(24, 14))

for ax, (name, model) in zip(axs.flatten(), models):
    model.fit(X_train, y_train)
    models_dict[name] = model
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
    mse_scores = np.sqrt(-scores)
    r2_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="r2")
    results.append((name, np.mean(mse_scores), np.mean(r2_scores)))

    #
    plot_learning_curve(ax, model, name, X_train, y_train, cv=10)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=300)
plt.show()

results_df = pd.DataFrame(results, columns=["Model", "MSE", "R²"])

best_model_name = max(results, key=lambda x: x[2])[0]
best_model = models_dict[best_model_name]

print("Best Model:", best_model_name)
print("RMSE:", results_df.loc[results_df['Model'] == best_model_name, 'MSE'].values[0])
print("R²:", results_df.loc[results_df['Model'] == best_model_name, 'R²'].values[0])

#
new_data = pd.read_csv(r"E:\Model\dataset.csv")

#
y_pred = best_model.predict(new_data)

#
new_data['predicted_y'] = y_pred

#
top_10 = new_data.nlargest(10, 'predicted_y')
print("Top 10 Predictions:")
print(top_10)

#
print("Max Predicted y:", top_10['predicted_y'].max())

#
max_predicted_y = new_data['predicted_y'].max()
print("Max Predicted y:", max_predicted_y)

rows_with_max_y = new_data[new_data['predicted_y'] == max_predicted_y]

#
unique_temp_rows = rows_with_max_y.sort_values('Molarity').drop_duplicates('Temp')

print("Rows with Max Predicted y and unique temp:")
print(unique_temp_rows)

#
negative_predictions = new_data[new_data['predicted_y'] < 0]

#
bottom_10 = negative_predictions.nsmallest(50, 'predicted_y')
print("Bottom 10 Predictions:")
print(bottom_10)

#
print("Min Predicted y:", bottom_10['predicted_y'].min())

#
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_model.predict(X_test),color="#3D7E7D")

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')

plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel('True Values', fontsize=28)
plt.ylabel('Predictions', fontsize=28)

plt.show()

#
print("Max Predicted y:", top_10['predicted_y'].max())
#
print("Min Predicted y:", bottom_10['predicted_y'].min())

#
try:
    importances = best_model.feature_importances_
except AttributeError:
    print("The best model doesn't support feature importances.")
else:
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:2]
    top_features = X.columns[top_indices]

    #
    new_data['feature_1'] = new_data.iloc[:, top_indices[0]]
    new_data['feature_2'] = new_data.iloc[:, top_indices[1]]
    new_data = new_data.sort_values(by='predicted_y', ascending=False)
    new_data = new_data.drop_duplicates(subset=['feature_1', 'feature_2'], keep='first')

    plt.figure(figsize=(24, 16))

    heatmap_data = new_data.pivot(index='feature_1', columns='feature_2', values='predicted_y')

    #
    ax = sns.heatmap(heatmap_data, cmap='crest', annot=False, fmt=".2f", annot_kws={"size": 24})
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=40)
    #
    #
    ax.tick_params(axis='x', labelsize=60)  #
    ax.tick_params(axis='y', labelsize=60)  #

    #
    ax.set_xlabel("Temperature (℃)", fontsize=70)
    ax.set_ylabel("Molarity (mM)", fontsize=70)

    #
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    #
    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()

# 
for i, label in enumerate(xlabels):
    if i % 2 != 0:
        label.set_visible(False)
for i, label in enumerate(ylabels):
    if i % 5 != 0:
        label.set_visible(False)
plt.savefig("model_25", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()