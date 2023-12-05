from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.model')

app = Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['GET', 'POST'])
def index ():
    prediction = None
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        BloodPressure = request.form['BloodPressure']
        Glucose = request.form['Glucose']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        test = [int(Pregnancies), int(BloodPressure), int(Glucose), int(SkinThickness), int(Insulin), int(BMI), int(DiabetesPedigreeFunction), int(Age)]
        test = np.array(test)
        print(test.shape)
        # prediction = model.predict(test)
        # print(prediction)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)
