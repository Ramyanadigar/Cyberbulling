from flask import Flask, render_template, request, redirect, url_for
from prediction import cyberBullyingLocation
import numpy as np
import pandas as pd
import shutil
import os

app = Flask(__name__)
app.secret_key = 'cyberbullying'


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            os.remove("static/input_file/selected_file.csv")
            selected_file = request.files['option'].filename
            shutil.copy(src=os.path.join("user_input", f"{selected_file}"), dst=os.path.join("static", "input_file", "selected_file.csv"))

            prdictedResult = cyberBullyingLocation()

            ip_data = read_csv("static/input_file/selected_file.csv")
            ip_title = "List of Input Data"

            re_data = read_csv("static/result_file/result.csv")
            re_title = "List of Reuslt Data"

            return render_template('index.html', pr=prdictedResult, i_t=ip_title, i_cols=ip_data[0], i_values=ip_data[1], r_t=re_title, r_cols=re_data[0], r_values=re_data[1])
        except Exception as e:
            print("An error occurred:", str(e))
            message = "Error occurred while sharing file."

        return render_template('index.html', msg=message)
    return render_template('index.html')


def read_csv(file):
    df = pd.read_csv(file)
    cols = list(df.columns)
    df1 = np.asarray(df)
    length = len(df1)
    df2 = []
    count = length
    for i in range(length):
        df2.append(df1[count - 1])
        count -= 1
    print("df2: ", df2)
    return cols, df2


@app.route("/home")
def home():
    return render_template("index.html")


@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):

                return redirect(url_for('home'))
        else:
            msg = 'Invalid Login Try Again'
            return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['a_name']
        email = request.form['a_email']
        password = request.form['a_password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': password}
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        # print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successfull !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('register.html')


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        email = request.form['a_email']
        current_pass = request.form['currentpsd']
        new_pass = request.form['newpsd']
        verify_pass = request.form['reenterpsd']

        if not email or not current_pass or not new_pass or not verify_pass:
            msg = 'Please fill in all fields'
            return render_template('password.html', msg=msg)

        r1 = pd.read_excel('user.xlsx')

        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    # Hash and store the new password securely in a real-world application
                    r1.loc[index, "password"] = str(verify_pass)
                    r1.to_excel("user.xlsx", index=False)
                    msg = 'Password changed successfully'
                    return render_template('changepsd.html', msg=msg)
                else:
                    msg = 'Re-entered password does not match'
                    return render_template('changepsd.html', msg=msg)

        msg = 'Incorrect email or password'
        return render_template('changepsd.html', msg=msg)

    return render_template('changepsd.html')


@app.route('/graph', methods=['POST', 'GET'])
def graph():
    try:
        if request.method == 'POST':
            graph_name = request.form['text']
            graph = ''
            name = ''

            if graph_name == "c_ac":            
                model_name = "Convolutional Neural Network"
                name = "Accuracy Plot Graph "
                graph = "static/graphs/c_ac.png"
            elif graph_name == 'c_ls':
                model_name = "Convolutional Neural Network"
                name = "Loss Plor Graph"
                graph = "static/graphs/c_ls.png"
            elif graph_name == 'c_cr':
                model_name = "Convolutional Neural Network"
                name = "Classification Report"
                graph = "static/graphs/c_cr.png"
            elif graph_name == 'c_cm':
                model_name = "Convolutional Neural Network"
                name = "Confusion Matrix"
                graph = "static/graphs/c_cm.png"

            return render_template('graphs.html', mn=model_name, name=name, graph=graph)
    except Exception as e:
         msg = "Select the Graph."
         return render_template('graphs.html', msg=msg)
    
    
@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('graphs.html')


if __name__ == '__main__':
    app.run(debug=True, port=4006, host='0.0.0.0')
