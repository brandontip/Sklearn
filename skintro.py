import json

file_name = "books_small.json"

class Review:
    def __init__(self,text,score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sent()
        
    def get_sent(self):
        if self.score <= 2: 
            return Sentiment.NEGATIVE
        elif self.score == 3: 
            return Sentiment.NEUTRAL
        elif self.score >= 3: 
            return Sentiment.POSITIVE
        
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    
# Load data    

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        review["reviewText"]
        review["overall"]
        reviews.append(Review(review["reviewText"],review["overall"]))
        
# Split data

from sklearn.model_selection import train_test_split
training, test = train_test_split(reviews, test_size = .33, random_state = 42)
train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]
test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]

# Bags of words vectorization

from sklearn.feature_extraction.text import CountVectorizer as CV
vectorizer = CV()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x) #just transform for test data

# Classification
# there are many different classification methods

from sklearn import svm
clf_svm = svm.SVC(kernel = "linear") # linear svm
clf_svm.fit(train_x_vectors, train_y)

from sklearn.tree import DecisionTreeClassifier # decision tree
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

from sklearn.naive_bayes import GaussianNB #Naive Bayes
clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.toarray(), train_y)

from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression()
clf_log.fit(train_x_vectors.toarray(), train_y)


# Evaluation

# Mean accuracy
clf_svm.score(test_x_vectors,test_y) # 82.4
clf_dec.score(test_x_vectors,test_y) # 75.7
clf_gnb.score(test_x_vectors.toarray(),test_y) # 81.2
clf_log.score(test_x_vectors.toarray(),test_y) # 83.0

# F1 Score
from sklearn.metrics import f1_score
f1_score(test_y, clf_svm.predict(test_x_vectors), average = None, labels = [Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE])
# 91.3 21.0 22.2 fscores

# all models performing poorly with negative/neutral indicates possible data issue
# further inspection shows a ton of bias towards positive labels in train data
# see intro2

    


