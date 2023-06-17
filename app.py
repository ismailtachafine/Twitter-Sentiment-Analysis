#importing the needed libs
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle,nltk,string,re
from nltk.corpus import stopwords
from random import randrange
# cleaning the texts
# importing the libraries for Natural Language Processing

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


#***********************************************************************#

#let's create a flask app
app=Flask(__name__)

#the flask app varibales
##news_title=""
news_text=""
color="black"


#***********************************************************************#


#stopward set
nltk.download('stopwords')
stop = set(stopwords.words('english'))

#les punctuations
punctuations=string.punctuation

#loading the example dataset
data = pd.read_csv('twitter.csv')

data = data[['Sentiment','Tweet content']]
data.columns = ['Sentiment', 'Review']


#load the pickle model
DT_model=pickle.load(open('extratrees.pkl','rb'))

#loading the count vectorizer to transform the text into numeric form
count_vectorizer=pickle.load(open('vectorizer.pkl','rb'))

#***********************************************************************#



#clean text function
def text_cleaned(review):
  corpus = []
  for i in range(0, len(data)):
    review = re.sub('@\S+', ' ', review)
    review = re.sub('http\S+', ' ', review)
    review = re.sub('\$\S+', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
  return str(review)



#***********************************************************************#

#this functoin will return the predicted valued for our target variable which is 0 or 1 or -1
def predictor(to_predict_list):
  to_predict = np.array(to_predict_list)
  #the result of prediction
  result = DT_model.predict(to_predict)
  print(result)
  return result[0]

@app.route("/")
def Home():
	return render_template("index.html")


@app.route("/example", methods=["GET"])
def example():
    index=randrange(0,len(data)-1,1)
    ##news_title=data.loc[index]["title"]
    news_text=data.iloc[index]["Review"]
    return render_template("index.html",news_text=news_text)

@app.route("/predict",methods=["POST"])
def predict():
  #on clicking on the predict button
  if request.method=='POST' :
    #getting the title from the form
    ##title=request.form["news_title"]
  
    #getting the text from the form
    text=request.form["news_text"]
  
    #concatinating the title with the text
    article_content=text

    #cleaning the article_content
    article_content=text_cleaned(article_content)

    #saving the result into a list
    to_predict_list=[article_content]

    #converting the text into a numeric vector
    to_predict_list=count_vectorizer.transform(to_predict_list).toarray()

    #getting the result of prediction
    result=predictor(to_predict_list)
    
    #checking the result if 1 then the news are fake else then the news are real depending on the given text
    if result=='Negative':
        color="red"
        prediction = ' Prediction : Negative ! '
    elif result=='Neutral':
        color="white"
        prediction = " Prediction : Neutral ! "
    elif result=='Irrelevant':
        color="blue"
        prediction = " Prediction : Irrelevant ! "
    else:
        color="green"
        prediction = " Prediction : Positive ! "
  return render_template("index.html",clean_text_label="Clean Review:",clean_text=article_content, prediction = prediction,p_color=color,news_text=news_text)


#***********************************************************************#


if __name__=="__main__":
  app.run(debug=True)