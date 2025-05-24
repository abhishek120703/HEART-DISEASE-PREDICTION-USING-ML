import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Download heart disease dataset
url = "heart.csv"
data = pd.read_csv(url)

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
