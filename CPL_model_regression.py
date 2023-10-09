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

data = pd.read_csv(r"E:\Model\data.csv")

X = data.iloc[:, 0:6]
y = data.iloc[:, 6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ("DT", DecisionTreeRegressor(random_state=42)),
    ("RF", RandomForestRegressor(random_state=42)),
    ("KNN", KNeighborsRegressor()),
    ("XGBoost", XGBRegressor(random_state=42)),
    ("CatBoost", CatBoostRegressor(random_state=42)),
    ("LightGBM", lgb.LGBMRegressor(random_state=42)),
]

results = []
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
    mse_scores = np.sqrt(-scores)
    r2_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="r2")
    results.append((name, np.mean(mse_scores), np.mean(r2_scores)))


results_df = pd.DataFrame(results, columns=["Model", "MSE", "R²"])

#
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="R²", data=results_df, palette="crest_r")

plt.xlabel("Models", fontsize=28)
plt.ylabel("R²", fontsize=28)
#plt.title("Model Performance Comparison", fontsize=14)

plt.xticks(rotation=45, fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("model_6", dpi=300, bbox_inches='tight')
plt.show()

#
plt.figure(figsize=(8, 6))
sns.barplot(x="Model", y="MSE", data=results_df, palette="crest_r")

plt.xlabel("Model", fontsize=16)
plt.ylabel("RMSE", fontsize=16)
#plt.title("Model Performance Comparison (RMSE)", fontsize=14)
plt.xticks(rotation=0, fontsize=14)
plt.yticks(np.arange(0,0.03, 0.005), fontsize=14)

#
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
plt.gca().yaxis.set_major_formatter(formatter)

plt.ylim(0, 0.02)

#plt.savefig("model_7", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

best_model = max(results, key=lambda x: x[2])

print("Best Model:", best_model[0])
print("RMSE:", best_model[1])
print("R²:", best_model[2])

for i, (name, model) in enumerate(models):
    fig, ax = plt.subplots(figsize=(8, 6))

    #
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    ax.scatter(y_train, y_train_pred, color='#3D7E7D', label='Train')
    ax.scatter(y_test, y_test_pred, color='#FDA056', label='Test')
    ax.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='black', linestyle='--')
    ax.set_xlabel('Actual', fontsize=28)
    ax.set_ylabel('Predicted', fontsize=28)
    ax.tick_params(axis='both', labelsize=24)
    #
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.set_title(f"{name}", fontsize=24)
    ax.legend(frameon=False, fontsize=24)

    plt.tight_layout()
    plt.savefig(f"modelL_{i+1}.png", dpi=300, bbox_inches='tight')
    plt.show()


X = data.drop('glum', axis=1)

correlation_matrix = X.corr()

correlation_matrix.columns = correlation_matrix.columns.str.replace('H2O', 'H$_2$O')
correlation_matrix.index = correlation_matrix.index.str.replace('H2O', 'H$_2$O')

fig = plt.figure(figsize=(12, 10))
ax = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='Oranges', annot_kws={'size': 24})

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=20)

plt.xticks(rotation=45, fontsize=28)
plt.yticks(rotation=45, fontsize=28)

plt.tight_layout()
plt.show()

