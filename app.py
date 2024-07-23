from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    if file:
        df = pd.read_csv(file)
        df.drop(['UDI','Product ID'], axis=1, inplace=True)
        df = pd.get_dummies(df, dtype=int)
        df.drop('Failure Type_No Failure', axis=1, inplace=True)
        df.rename(columns={"Target":"Failure"}, inplace=True)

        x_cols = ['Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]',
          'Torque [Nm]', 'Tool wear [min]','Type_H','Type_L', 'Type_M']
        y_cols = ['Failure']
        x = df[x_cols]
        y = df[y_cols]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True, random_state=42)

        train_indices = x_train.index
        test_indices = x_test.index

        model = RandomForestClassifier(random_state=42)
        model.fit(x_train, y_train.values.ravel())
        # y_pred_prob = model.predict_proba(x_test)

        y_cols_failure_type = ['Failure Type_Heat Dissipation Failure', 'Failure Type_Overstrain Failure', 'Failure Type_Power Failure', 'Failure Type_Random Failures', 'Failure Type_Tool Wear Failure']
        y_failure_type = df[y_cols_failure_type]

        x_train2 = x.loc[train_indices]
        x_test2 = x.loc[test_indices]
        y_train2 = y_failure_type.loc[train_indices]
        y_test2 = y_failure_type.loc[test_indices]

        model2 = RandomForestClassifier(random_state=42)
        model2.fit(x_train2, y_train2)

        # y_pred_fail_type = model2.predict(x_test2)

        joblib.dump(model, 'failure_prob.pkl')
        joblib.dump(model2, 'model_failure_type.pkl')
        
        return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    pass # Here we will write function which takes single data about machine's current state as input from frontend and then predict failure using model pickle file saved in previous route  

if __name__ == '__main__':
    app.run(debug=True)