from flask import Flask, render_template,url_for, request
import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import re 
import emoji
import pyarabic.araby as araby
from nltk.corpus import stopwords
import nltk
from nltk.stem.isri import ISRIStemmer
import tweepy
from tweepy import API
from tweepy import OAuthHandler
from flask_caching import Cache
import credtional_keys

#---------------------------------------------------------------------------
auth = tweepy.OAuthHandler(credtional_keys.TWITTER_APP_KEY,credtional_keys.TWITTER_APP_SECERT)
auth.set_access_token(credtional_keys.TWITTER_KEY,credtional_keys.TWITTER_SECERT)
api=tweepy.API(auth)
#-------------------------------------------------------------------------
cache = Cache(config={'CACHE_TYPE': 'simple'})
app = Flask(__name__)
cache.init_app(app)

@app.route('/')
@cache.cached(timeout=300)
def index():
	return render_template('index.html')


@app.route('/detect',methods=['POST'])
def detect():
	try:
		#load Dataset
		train=pd.read_csv("/Users/Lina/Desktop/depressionDetector/data/newVersion-2Cleaned.csv", encoding='utf-8', delimiter=',')
		Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train["Tweets"],train["Label"],random_state=1,test_size=0.20)
		#TFIDF
		vectorizerTFIDF = TfidfVectorizer(max_features=450, decode_error="ignore")
		vectorizer = vectorizerTFIDF.fit_transform(Train_X)
		#load BNB model
		with open("/Users/Rahaf/Desktop/depressionDetector/data/BNB_model.pkl", 'rb') as file:
			clf = pickle.load(file)

		if request.method=='POST':
			if request.form['comment']=='':
				return render_template('index.html')
			comment = request.form['comment']
			try:
				user_tweets=get_tweets(api,comment) #Get username's list of tweets
			except tweepy.TweepError:
				error_message="Invalid Username"
				return render_template('index.html',result=error_message)
			data = user_tweets["Tweets"] #Select just Tweets column
			if data.empty:
				return render_template('index.html',result="There is No recent tweets of " + str(comment))	
			cleaned_data=data.apply(lambda x:pre_processing(x))
			vect= vectorizerTFIDF.transform(cleaned_data).toarray()
			predoctionList = clf.predict(vect)
			my_prediction=get_finalPrediction(predoctionList)
			return render_template('index.html',result="About " + str(my_prediction) + "%"+" of "+ str(comment)+" recent tweets indicate depression")
	except ValueError:
		return render_template('index.html')
		
def get_tweets(api,user):
    TweetsFinal=[]
    tweets =api.user_timeline (screen_name=user, count=200 , exclude_replies=True, include_rts = False)
    for tweet in tweets:
        if((not tweet.entities.get('media',[])) and (not tweet.entities.get('urls',[]))):
            TweetsFinal.append([tweet.text])
                
    T_dataframe = pd.DataFrame(TweetsFinal,columns = ["Tweets"])
    return T_dataframe

def get_finalPrediction (predoctionList):
	depressed=0
	not_depressed=0
	list_length=len(predoctionList)
	for a in predoctionList:
		if a==1:depressed=depressed+1
		else:not_depressed=not_depressed+1
    
	percentage=(depressed/list_length)*100
	return(round(percentage,2))


def pre_processing(tweet):
	Arabic_numbers = ['٤','١','٢','٣','٥','٦','٧','٨','٩']
	special_character = ['؟','،','?',',','!','.',':','"','""','‘‘','‘','؛',
             		     '@','#','$','%','^','&','*','()',')','(','\\','/','((', '_', '"','"']
    
	emoticons = [':-)', ':)', ';)', ':o)',':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
                 ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D'':L',
                 ':-/', '>:/', ':S', '>:[',':@',':(','>.<',';(',':c', ':{',':<',':")','):',':-[', ':-<',
                 ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)','X-P','x-p', 'xp', 'XP']
	tweet = emoji.get_emoji_regexp().sub(u'', tweet)
    
	for word in range(0, len(special_character)):
		tweet = tweet.replace(special_character[word], '') 
    
	for word in range(0, len(emoticons)):
  		tweet = tweet.replace(emoticons[word], '') 
     
	for word in range(0, len(Arabic_numbers)):
		tweet = tweet.replace(Arabic_numbers[word], '') 

	tweet = re.sub(r'[0-9a-zA-Z]+',' ', tweet)
	tweet = re.sub("[إأٱآا]", "ا", tweet)
	tweet = re.sub("ى", "ي", tweet)
	tweet = re.sub("ة", "ه", tweet)
	tweet = re.sub(r'(.)\1+', r'\1', tweet)
	tweet = re.sub(r'\s+', ' ', tweet)   
	tweet = araby.strip_tashkeel(tweet)
	tweet = araby.strip_tatweel(tweet)
	stop_words = set(stopwords.words("arabic"))
	words = araby.tokenize(tweet)
	tweet = " ".join([w for w in words if not w in stop_words])
	return tweet



if __name__ == '__main__':
	app.run(debug=True)
