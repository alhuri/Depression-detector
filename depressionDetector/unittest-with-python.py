from app import get_tweets,get_finalPrediction,pre_processing,api
import unittest
import pandas as pd 

class testApp(unittest.TestCase):
    
    def setUp(self):
        self.user="ccse_tu"
        self.tweet="أعاني من اكتئاب حاد"
        self.predoctionList={2,1,1,1,2}

    def test_get_tweets(self):
        resp=get_tweets(api,self.user)
        self.assertIsInstance(resp,pd.DataFrame)

    def test_pre_processing(self):
        resp=pre_processing(self.tweet)
        self.assertIsInstance(resp,str)

    def test_get_finalPrediction(self):
        resp=get_finalPrediction(self.predoctionList)
        self.assertIsInstance(resp,float)

if __name__ == "__main__":
    unittest.main()