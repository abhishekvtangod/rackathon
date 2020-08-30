import os

from flask import Flask, render_template, request

#from ocr import ocr_core

from stegno import *

UPLOAD_FOLDER = '/static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/steg_en', methods=['POST','GET'])
def stag_encode():
    if request.method =='POST':   
        text = request.form['paragraph_text']
        imag = request.files['image']
        extracted_text = encode(imag,text)
        return render_template('upload.html',
                           msg='Successfully processed')
    elif request.method =='GET':
        return render_template('upload.html')


@app.route('/steg_dec', methods=['GET','POST'])
def stag_decode():
    if request.method == 'POST':
        imag = request.files['image']
        # # check if the post request has the file part
        # if 'file' not in request.files:
        #     return render_template('upload.html', msg='No file selected')
        # file = request.form['image']
        # # if user does not select file, browser also
        # # submit a empty part without filename
        # if file.filename == '':
        #     return render_template('upload.html', msg='No file selected')

        # if file and allowed_file(file.filename):
        #     file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))

        #     # call the OCR function on it
        extracted_text = decode(imag)

            # extract the text and display it
        return render_template('ans.html',
                                   extracted_text=extracted_text)
    #elif request.method == 'GET':
    return render_template('upload_dec.html')


if __name__ == '__main__':
    app.run(debug = True)