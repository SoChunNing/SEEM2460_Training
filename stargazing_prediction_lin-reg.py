import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import os
import sys
import seaborn as sns
from datetime import datetime

def time_string_to_number(time_string):
    """
    Convert a time string in the format 'HH:MM' to a number of minutes since midnight.
    """
    time_only = datetime.strptime(time_string, "%H:%M").time()
    return time_only.hour * 60 + time_only.minute

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))                          
data = pd.read_csv(f'{script_dir}/nsb_weather_merged.csv')
data = data.dropna()
data = data.drop(columns=['Date'])
#Convert time strings to numbers
data['Sun Rise'] = data['Sun Rise'].apply(time_string_to_number)
data['Sun Transit'] = data['Sun Transit'].apply(time_string_to_number)
data['Sun Set'] = data['Sun Set'].apply(time_string_to_number)
data['Moon Rise'] = data['Moon Rise'].apply(time_string_to_number)
data['Moon Transit'] = data['Moon Transit'].apply(time_string_to_number)
data['Moon Set'] = data['Moon Set'].apply(time_string_to_number)
y = data[['Max Night Sky Brightness (MPSAS)', 'Min Night Sky Brightness (Non-zero) (MPSAS)', 'Mean Night Sky Brightness (Excluded zero) (MPSAS)']]
X = data.iloc[:, 3:]

#Random forest regression
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit on training set
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['Max Night Sky Brightness (MPSAS)', 'Min Night Sky Brightness (Non-zero) (MPSAS)', 'Mean Night Sky Brightness (Excluded zero) (MPSAS)'])
y_test = pd.DataFrame(y_test, columns=['Max Night Sky Brightness (MPSAS)', 'Min Night Sky Brightness (Non-zero) (MPSAS)', 'Mean Night Sky Brightness (Excluded zero) (MPSAS)'])

#Evaluate model performance
print("Model training complete.")
print("Evaluating model performance...")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R2):", r2_score(y_test, y_pred))
print("Cross-validation scores:", cross_val_score(model, X_train, y_train, cv=5))
print("Saving model...")
joblib.dump(model, f'{script_dir}/stargazing_model_lin-reg.pkl')

#Plot correlation heatmap
corr = data.corr()
plt.figure(figsize=(20, 16)) 
sns.heatmap(
    corr,                            # the correlation matrix
    xticklabels=corr.columns,        # use column names on x‑axis
    yticklabels=corr.columns,        # use column names on y‑axis
    annot=True,                      # write numeric value in each cell
    fmt=".2f",                       # format annotations to 2 decimal places
    cmap="coolwarm",                 # diverging color palette
    linewidths=0.5                   # lines between cells
)
plt.xticks(rotation=45, ha='right')
plt.savefig(f'{script_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.title("Correlation Heatmap")    
plt.tight_layout()
plt.show()

#Plot coefficient importance
coefs = np.abs(model.coef_[0])  
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'x{i}' for i in range(X_train.shape[1])]
coef_df = pd.DataFrame(coefs, index=feature_names, columns=['Coefficient'])
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(coef_df.index, coef_df['Coefficient'], color='skyblue', align="center")
plt.ylabel('Coefficient Value')
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{script_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()