import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import sys
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))                          
data = pd.read_csv(f'{script_dir}/nsb_weather_merged.csv')
data = data.drop(columns=['Date'])
y = data[['Max Night Sky Brightness (MPSAS)', 'Min Night Sky Brightness (Non-zero) (MPSAS)', 'Mean Night Sky Brightness (Excluded zero) (MPSAS)']]
X = data.iloc[:, 3:]

#Random forest regression
model = RandomForestRegressor(n_estimators=100, random_state=42, criterion='squared_error')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Fit on training set
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['max', 'non_zero_min', 'non_zero_mean'])
y_test = pd.DataFrame(y_test, columns=['max', 'non_zero_min', 'non_zero_mean'])

#Evaluate model performance
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Model accuracy: ", model.score(X_test, y_test))
print("Model training complete.")
print("Saving model...")
joblib.dump(model, f'{script_dir}/stargazing_model.pkl')

#Plot correlation heatmap
corr = data.corr()
plt.figure(figsize=(10, 8)) 
sns.heatmap(
    corr,                            # the correlation matrix
    xticklabels=corr.columns,        # use column names on x‑axis
    yticklabels=corr.columns,        # use column names on y‑axis
    annot=True,                      # write numeric value in each cell
    fmt=".2f",                       # format annotations to 2 decimal places
    cmap="coolwarm",                 # diverging color palette
    linewidths=0.5                   # lines between cells
)
plt.savefig(f'{script_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.title("Correlation Heatmap")    
plt.tight_layout()
plt.show()