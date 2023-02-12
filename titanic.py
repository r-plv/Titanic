from flask import Flask,redirect,request,render_template
import pickle
import numpy as np

titanic = Flask(__name__)

@titanic.route("/")
def fun1():
    return render_template("titanic.html")

@titanic.route("/predict", methods=["post"])
def fun2():
    PassengerId = request.form['PassengerId']
    Pclass = request.form['Pclass']
    Name = request.form['Name']
    Sex = int(request.form['Sex'])
    Age = request.form['Age']
    SibSp = request.form['SibSp']
    Parch = request.form['Parch']
    Ticket = request.form['Ticket']
    Fare = request.form['Fare']
    Cabin = request.form['Cabin']
    Embarked = request.form['Embarked']
    mymodel = pickle.load(open('titanic.pkl',"rb"))
    Survived = mymodel.predict([[PassengerId,Pclass,Sex,Age,SibSp,Parch,Cabin,Fare,Embarked]])[0]

    if Survived == 0:
        out = "Not Survived"
    else:
        out = "Survived"

    return "{} has {}".format(Name,out)

if __name__=="__main__":
    titanic.run(debug=True)