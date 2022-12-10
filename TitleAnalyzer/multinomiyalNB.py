import numpy as np
import pandas as pd
import nltk
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords

data = pd.read_csv('/Users/macintosh/Desktop/TitleAnalyzer/data/book30-listing-train.csv',encoding = "ISO-8859-1", header=None)
print(data.shape)
columns = ['Id', 'Image', 'Image_link', 'Title', 'Author', 'Class', 'Genre']
data.columns = columns
books = pd.DataFrame(data['Title'])
genre = pd.DataFrame(data['Genre'])

feat = ['Genre']
for x in feat:
    le = LabelEncoder()
    le.fit(list(genre[x].values))
    genre[x] = le.transform(list(genre[x]))

def change(t):
    t = t.split()
    return ' '.join([(i) for (i) in t if i not in stop])
stop = list(stopwords.words('english'))
data['Title'].apply(change)

vectorizer = TfidfVectorizer(max_features=70000, strip_accents='unicode',lowercase=True,
                            analyzer='word', token_pattern=r'\w+', use_idf=True, 
                            smooth_idf=True, sublinear_tf=True, stop_words = 'english')
vectors = vectorizer.fit_transform(data['Title'])

X_train, X_valid, y_train, y_valid = train_test_split(vectors, genre['Genre'], test_size=5700)

clf = MultinomialNB(alpha=.45)
clf.fit(X_train, y_train)
pred = clf.predict(X_valid)
accuracy = metrics.accuracy_score(y_valid, pred)
print('Accuracy of the model: ', accuracy)

joblib.dump(clf, '/Users/macintosh/Desktop/TitleAnalyzer/models/multinomialNB.pkl')
joblib.dump(vectorizer, '/Users/macintosh/Desktop/TitleAnalyzer/models/vectorizer.sav')

