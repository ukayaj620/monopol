from flask import Flask, request
from flask.templating import render_template

app = Flask(__name__)


@app.route('/')
def index():
    if request.method == 'POST':
        return render_template('index.html', preprocessed_image="./data/example/indonesia_nopol_1.JPG", classified_text="BK1689UZ")

    return render_template('index.html')
