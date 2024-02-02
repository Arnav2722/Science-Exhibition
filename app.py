from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)


def predict(values, dic):
    dictonary = {
        "rbc": {
            "abnormal": 1,
            "normal": 0,
        },
        "pc": {
            "abnormal": 1,
            "normal": 0,
        },
        "pcc": {
            "present": 1,
            "notpresent": 0,
        },
        "ba": {
            "notpresent": 0,
            "present": 1,
        },
        "htn": {
            "yes": 1,
            "no": 0,
        },
        "dm": {
            "yes": 1,
            "no": 0,
        },
        "cad": {
            "yes": 1,
            "no": 0,
        },
        "appet": {
            "good": 1,
            "poor": 0,
        },
        "pe": {
            "yes": 1,
            "no": 0,
        },
        "ane": {
            "yes": 1,
            "no": 0,
        },
    }
    for i in range(len(values)):
        if values[i] in dictonary[dic[i]]:
            values[i] = dictonary[dic[i]][values[i]]
    if len(values) == 8:
        model = pickle.load(open("models/diabetes.pkl", "rb"))
    elif len(values) == 26:
        model = pickle.load(open("models/breast_cancer.pkl", "rb"))
    elif len(values) == 13:
        model = pickle.load(open("models/heart.pkl", "rb"))
    elif len(values) == 18:
        model = pickle.load(open("models/kidney.pkl", "rb"))
    elif len(values) == 10:
        model = pickle.load(open("models/liver.pkl", "rb"))

    values = np.asarray(values, dtype=float)
    return model.predict(values.reshape(1, -1))[0]


@app.route("/")
def home():
    return render_template("index1.html")


@app.route("/contactus")
def contactUs():
    return render_template("contact.html")


@app.route("/aboutus")
def aboutUs():
    return render_template("about-us.html")


@app.route("/diabetes", methods=["GET", "POST"])
def diabetesPage():
    return render_template("diabetes1.html")


@app.route("/cancer", methods=["GET", "POST"])
def cancerPage():
    return render_template("breast_cancer1.html")


@app.route("/heart", methods=["GET", "POST"])
def heartPage():
    return render_template("heart1.html")


@app.route("/kidney", methods=["GET", "POST"])
def kidneyPage():
    return render_template("kidney1.html")


@app.route("/liver", methods=["GET", "POST"])
def liverPage():
    return render_template("liver1.html")


@app.route("/malaria", methods=["GET", "POST"])
def malariaPage():
    return render_template("malaria1.html")


@app.route("/predict", methods=["POST", "GET"])
def predictPage():
    try:
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("index1.html", message=message)

    return render_template("predict.html", pred=pred)


@app.route("/breastCancerPredict", methods=["POST", "GET"])
def cancerPredictPage():
    try:
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("index1.html", message=message)

    return render_template("breast_cancer_predict.html", pred=pred)


@app.route("/diabetesPredict", methods=["POST", "GET"])
def diabetesPredictPage():
    try:
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("index1.html", message=message)

    return render_template("diabetes_predict.html", pred=pred)


@app.route("/heartPredict", methods=["POST", "GET"])
def heartPredictPage():
    try:
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("index1.html", message=message)

    return render_template("heart_predict.html", pred=pred)


@app.route("/kidneyPredict", methods=["POST", "GET"])
def kidneyPredictPage():
    try:
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            to_predict_list = []
            for val in list(to_predict_dict.values()):
                try:
                    to_predict_list.append(float(val))
                except ValueError:
                    message = f"Invalid value: {val}. Please enter valid data."
                    return render_template("index1.html", message=message)
            pred = predict(to_predict_list, to_predict_dict)
    except Exception as e:
        message = f"An error occurred: {str(e)}"
        return render_template("index1.html", message=message)

    return render_template("kidney_predict.html", pred=pred)


@app.route("/liverPredict", methods=["POST", "GET"])
def liverPredictPage():
    try:
        if request.method == "POST":
            to_predict_dict = request.form.to_dict()
            to_predict_list = []
            for val in list(to_predict_dict.values()):
                try:
                    to_predict_list.append(float(val))
                except ValueError:
                    message = f"Invalid value: {val}. Please enter valid data."
                    return render_template("index1.html", message=message)
            pred = predict(to_predict_list, to_predict_dict)
    except Exception as e:
        message = f"An error occurred: {str(e)}"
        return render_template("index1.html", message=message)

    return render_template("liver_predict.html", pred=pred)


@app.route("/malariapredict", methods=["POST", "GET"])
def malariaPredictPage():
    if request.method == "POST":
        try:
            if "image" in request.files:
                img = Image.open(request.files["image"])
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template("malaria1.html", message=message)
    return render_template("malaria1_predict.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True)
