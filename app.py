from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure model.pkl is in the same folder)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def team():
    return render_template('team.html')  # Team Page

@app.route('/')
def home():
    return render_template('index.html')  # Form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [float(request.form.get(feat)) for feat in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Prepare message and risk class
        if prediction == 1:
            result = "High risk of heart disease. Please consult a doctor."
            risk = "high"
        else:
            result = "Low risk of heart disease. Keep up the healthy lifestyle!"
            risk = "low"

        return render_template('result.html', prediction_text=result, risk=risk)
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

