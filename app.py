import os, sys, shutil, time
import pickle
from flask import Flask, request, jsonify, render_template,send_from_directory
from call import ocr
from werkzeug.utils import secure_filename
# from sklearn.externals import joblib
# from sklearn.ensemble import RandomForestClassifier

import urllib.request
import json
# from geopy.geocoders import Nominatim

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template,send_from_directory
from flask_mail import Mail,Message
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from flask_mail import Mail,Message
from stegno import *

UPLOAD_FOLDER = '/static/uploads/'
UPLOAD_FOLDER1 = '/static/uploads1/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



#from flask_restful import Api,Resource,reqparse

app = Flask(__name__)
#api=Api(app)
# app.config['MAIL_SERVER']='smtp.gmail.com'
# app.config['MAIL_PORT'] = 465
# app.config['MAIL_USERNAME'] = 'dig.sec.solutions@gmail.com'
# app.config['MAIL_PASSWORD'] = 'digsec101'
# app.config['MAIL_USE_TLS'] = False
# app.config['MAIL_USE_SSL'] = True

mail=Mail(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def func(x):

	


# #reading csv file
# 	print("......................Reading data......................")
# 	train = pd.read_csv('train.csv', encoding='ISO-8859-1')
# 	print("......................Data read......................")


# #preprocessing
# 	print("......................Starting Preprocessing......................")
# 	count_vect = CountVectorizer()
# 	X_train_counts = count_vect.fit_transform(train.text)
# 	tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
# 	X_train_tf = tf_transformer.transform(X_train_counts)


# #splitting dataset
# 	print("......................Starting training......................")
# 	y = train.author
# 	X_train_tf,y = shuffle(X_train_tf,y)
# 	X_train, X_test, y_train, y_test = train_test_split(X_train_tf, y, test_size=0.2)


# #MultiNB
# 	print("......................MultiNB......................")
# 	clfNB = MultinomialNB().fit(X_train, y_train)
# 	predictedNB = clfNB.predict(X_test)
# 	print(accuracy_score(y_test,predictedNB))


# #SVM Classifier
# 	print("......................SVM......................")
# 	clfSVM = SGDClassifier(loss='hinge',penalty='l2',
#                         alpha=1e-3, random_state=42,
#                         max_iter=5, tol=None).fit(X_train,y_train)
# 	predictedSVM = clfSVM.predict(X_test)
# 	print(accuracy_score(y_test,predictedSVM))

	with open('finalized_model.pkl', 'rb') as f:
		count_vect,tf_transformer,clfSVM = pickle.load(f)	


#Prediction
	doc=[]
	# str = 'nt it seems te me how much money is he worth and that within a reasonable margin i can tell you as for i private character most likely it is like that of other men which means that the less you investigate it th happier you will be private character indeed what have the men of to day to do with things miss made a pretty picture as she sat there in mrs s sitting room she was a pro of perhaps one and twenty i that s the of good height and with one of those well rounded figures which would better please did they not arouse fears of too great in the distant future at present she was as nearly perfect a as it it possible to imagine her mass of straw tinted hair was arranged in a manner that would have become a queen her head was poised on her she had a bright color thai was all her own and her excessive vivacity became her well mrs was a widow of about forty years of age who might once have been beautiful but who now seemed to feel the cares of life too heavily to mind the deep which time had hastened to place in the lines of her pale countenance she had a certain air of dignity as one who had seen better days and could rise in thought at least above the misfortunes of recent years the house which she occupied was situated within pistol shot of college in the city of cambridge it was a comfortable structure set back from the street and almost hidden from the gaze of by the shade trees and very tall hedge which bordered the her support came almost wholly from the letting of rooms to students from which source she managed to support in a modest manner herself and two daughters she listened with much interest to miss s and then said he is rich then you may set that down as assured when i have promised to marry him smiled the young lady his grandfather was one of those mill owners who made such a pile of money fifty years or so ago he left it all to when he was a baby that ia to his of and it has kept oc grow w t ay and growing as such fortunes do oh yes do is rich enough but as for his character that s quite thing i haven t got as far as that yet the speaker paused and looked at her companion if the troubles of this world had mad much impression on her young life there was no outward evidence of it a big yellow cat which sat on the window seat a few yards away seemed quite as much worried by either past or present as she yet this young lady had a history which would make the foundation of a romance an air of mystery pervaded her life which no one seemed able to penetrate she had her moods too and the of one of them seemed very little like the of another mrs had known her when a child her father captain arthur was a cambridge man and when her mother died mrs saw much of the pretty little orphan girl until the captain took her out to south america whence he never returned one of the first things did after coming back a grown young lady was to seek out mrs s house the old friendship was renewed and as is often the case it gained strength from the fact that the two ladies were so totally as well as from other causes which will appear later on i am glad indeed that you are to be so happy said the widow money is not a thing to be despised and those who pretend to so consider it art only striving for effect i have often felt the need of t since my husband died partly for myself but much for and money will do a great deal to make life pleasant but the first requisite in a mat partner should be private character art not cannot comprehend this fully you are fa young you remember henry there were thing about him which i could have wished different but ae one ever his private character when th bank officials found that he had fled with the books in such bad shape and thirty thousand dollars gone it was a blow to me the news which followed so soon of his death abroad was very hard to bear my only consolation in my distress was the reflection that whatever he might have done his private character was there was a momentary suspicion of amusement in miss s deep blue eyes but she mastered it before it attracted her friend s attention and said i am sorry to say i can t agree with you if try husband should ever run away he might take all the women in america with him for all of me but if he forgot to leave me a handsome pile of cash i d never forgive him never i have all right on that score in advance the day we are married he is to give me fifty thousand dollars that ll make a sure foundation for me in case anything should happen ah mrs if every husband would do as well how much more married happiness there would be miss laughed at her own and the elder lady s features relaxed a little and you have been engaged for mere than a year said mrs yes and i would make it ten if i is if dared for he might have heart disease and go off suddenly i could get him to give me the money now but that would have a disagreeable look i think would wait for me a century if i compelled ms i mob that s tub them sum '
	doc.append(str(x))
#print(doc)
	X_new_counts = count_vect.transform(doc)
	X_new_tfidf = tf_transformer.transform(X_new_counts)

	predicted = clfSVM.predict(X_new_tfidf)

	return predicted
	
	


'''*************************END*******************************'''

@app.route('/')
def root():
	
	return render_template('home_page.html')

@app.route('/text_audit',methods=['GET', 'POST'])
def ocr1():
    if request.method == 'POST':
        # check if the post request has the file part
        # if 'file' not in request.files:
        #     return render_template('image_audit.html', msg='No file selected')
        print(request.files['file'])
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        # if file.filename == '':
        #     return render_template('image_audit.html', msg='No file selected')

        # if file and allowed_file(file.filename):
        file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER1, file.filename))
        # print(file)
        # call the OCR function on it
        extracted_text = ocr(file)

        # extract the text and display it
        return render_template('form_pagev3.html',
                               msg='Coming from Image_Audit',
                               extracted_text=extracted_text,
                               img_src=UPLOAD_FOLDER1 + file.filename)
    elif request.method == 'GET':
        return render_template('form_pagev2.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')

        if file and allowed_file(file.filename):
            file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))
            print(file) 
            # call the OCR function on it
            extracted_text = ocr(file)

            # extract the text and display it
            return render_template('upload.html',
                                   msg='Successfully processed',
                                   extracted_text=extracted_text,
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template('upload.html')

@app.route('/steg')
def steg():
	return render_template('steg.html')

@app.route('/steg_en', methods=['POST','GET'])
def stag_encode():
    if request.method =='POST':   
        text = request.form['paragraph_text']
        imag = request.files['image']
        extracted_text = encode(imag,text)
        return render_template('steg_en.html',
                           msg='Successfully processed')
    elif request.method =='GET':
        return render_template('steg_en.html')



@app.route('/steg_dec', methods=['GET','POST'])
def steg_decode():
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
        return render_template('decoded_message.html',
                                   extracted_text=extracted_text)
    #elif request.method == 'GET':
    return render_template('steg_dec.html')

@app.route('/image_audit')
def index():
	return render_template('image_audit.html')

# @app.route('/images/<Paasbaan>')
# def download_file(Paasbaan):
# 	return send_from_directory(app.config['images'], Paasbaan)



# @app.route('/work.html')
# def work():
# 	return render_template('work.html')

# @app.route('/about.html')
# def about():
# 	return render_template('about.html')

# @app.route('/contact.html')
# def contact():
# 	return render_template('contact.html')

@app.route('/result', methods = ['POST'])
def predict():
	if request.method == 'POST':
		auth_id=request.form['auth_id']
		x=request.form['paragraph_text']

		

		
		
		
		y_res=str(func(x)[0])
		'''z1="Author : "+y_res
		z2="Possible Author : "+y_res	 
	'''
		print("**********************************")
		print("**********************************")
		print(y_res)
		print("**********************************")
		print("**********************************")
	
		
		if y_res == auth_id:
			my_prediction=['Verification Passed', 'tick.jpg', 'Accuracy : 81%', "Author: "+auth_id, '#A1E44D' ]
			#print("ROBBERY")
		else:
			#msg=Message('ALERT!!! Please Review',sender='dig.sec.solutions@gmail.com',recipients=['khanzaifa37@gmail.com'])
			#msg.body="Critical Security Alert. We have found that someone has tried to check your authorship (author-id)"+auth_id+"and failed. Please review this action. Thankyou, Digsec."
			#mail.send(msg)
			my_prediction=['Verification Failed', 'fail.jpg', "Probable Author :" + y_res, 'With Probability : 81%', '#C81D25' ]
			#print("SAFE")
		
		
	
	return render_template('result_pass.html', prediction = my_prediction)

# class exten(Resource):
# 	def get(self):
# 		return "Test Pass",200
# 	def post(self):
# 		parser=reqparse.RequestParser()
# 		parser.add_argument("auth_id")
# 		parser.add_argument("paragraph_text")
# 		params=parser.parse_args()

# 		ans=func(params["paragraph_text"])[0]
# 		if str(ans)==params["auth_id"]:
# 			return "Verification Pass",201
# 		else:
# 			return "Verification Failed",201

# api.add_resource(exten,"/test-api")

if __name__ == '__main__':
	app.run(debug = True)
