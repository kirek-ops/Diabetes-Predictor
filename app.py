from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

model = tf.keras.models.load_model('model.model')

app = Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['GET', 'POST'])
def index ():
    result = None
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        BloodPressure = request.form['BloodPressure']
        Glucose = request.form['Glucose']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        test = [float(Pregnancies), float(BloodPressure), float(Glucose), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        test = np.array(test)
        test = test.reshape(1, 8)
        test = scaler.transform(test)
        prediction = model.predict(test)
        if prediction < 0.5: 
            result = 0
        else:
            result = 1
        print(result)

    return render_template('index.html', data = result)

if __name__ == "__main__":
    app.run(debug = True)
