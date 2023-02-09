from flask import Flask, render_template, request
import pandas as pd
import pickle

app= Flask(__name__)
data=pd.read_csv("Cleaned_data.csv")
pipe= pickle.load(open("RidgeModel.pkl",'rb')) # rb refers to read-only mode
pipe2= pickle.load(open("LinearModel.pkl",'rb'))
pipe3= pickle.load(open("LassoModel.pkl",'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get("LOCATION")
    bhk = (float)(request.form.get("BHK"))
    bath = (float)(request.form.get("BATH"))
    sqft = request.form.get("SQFT")
    print(location, bath, bhk, sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction= formatINR(int(pipe.predict(input)[0]*100000))
    prediction2= formatINR(int(pipe2.predict(input)[0]*100000))
    prediction3= formatINR(int(pipe3.predict(input)[0]*100000))

    #return str(prediction)
    return render_template('outcome.html', prediction_text = prediction, prediction_text2 = prediction2, prediction_text3 = prediction3)
def formatINR(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
