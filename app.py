from flask import Flask, render_template, request
import tensorflow as tf

model = tf.keras.models.load_model('model.model')

app = Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['GET', 'POST'])
def index ():
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        print(Pregnancies, Glucose, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)
