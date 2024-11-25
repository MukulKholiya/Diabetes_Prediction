from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('diabetes_prediction.pkl', 'rb'))
labels = ['No', 'Yes']

# Store true and predicted values for metrics
y_true = []
y_pred = []

@app.route('/')
def home():
    return render_template('diabetes.html', pred=None)

@app.route('/predict', methods=['POST'])
def predict():
    global y_true, y_pred

    try:
        gender = float(request.form.get('gender'))
        age = float(request.form.get('age'))
        hypertension = float(request.form.get('hypertension'))
        heart_disease = float(request.form.get('heart_disease'))
        smoking_history = float(request.form.get('smoking_history'))
        HbA1c_level = float(request.form.get('HbA1c_level'))
        blood_glucose_level = float(request.form.get('blood_glucose_level'))

        inputs = np.array([[gender, age, hypertension, heart_disease, smoking_history, HbA1c_level, blood_glucose_level]])
        prediction = model.predict(inputs)[0]

        true_label = 0 if HbA1c_level < 6.0 and blood_glucose_level < 140 else 1
        y_true.append(true_label)
        y_pred.append(prediction)

        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()

        return render_template('diabetes.html', pred=f"Diabetes Prediction: {labels[prediction]}", precision=round(precision, 2), recall=round(recall, 2), accuracy=round(accuracy, 2), confusion_matrix=conf_matrix)

    except Exception as e:
        return render_template('diabetes.html', pred=f"Error: {str(e)}")

@app.route('/performance')
def performance():
    global y_true, y_pred
    precision = recall = accuracy = conf_matrix = None

    if y_true and y_pred:
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    
    return render_template('performance.html', precision=round(precision, 2) if precision is not None else None, recall=round(recall, 2) if recall is not None else None, accuracy=round(accuracy, 2) if accuracy is not None else None, confusion_matrix=conf_matrix if conf_matrix is not None else None)

if __name__ == '__main__':
    app.run(debug=True)
