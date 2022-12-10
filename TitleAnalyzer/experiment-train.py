import pandas as pd
import joblib
from numpy import savetxt

model = joblib.load('/Users/macintosh/Desktop/TitleAnalyzer/models/multinomialNB.pkl')
vec = joblib.load('/Users/macintosh/Desktop/TitleAnalyzer/models/vectorizer.sav')

data = pd.read_csv('/Users/macintosh/Desktop/TitleAnalyzer/data/book30-listing-test.csv', encoding = "ISO-8859-1", header=None)
#print(data.shape)
data.columns = ['Id', 'Image', 'Image_link', 'Title', 'Author', 'Class', 'Genre']

s = vec.transform(data['Title'])
probs = model.predict_proba(s)
print(probs.shape)

savetxt('/Users/macintosh/Desktop/TitleAnalyzer/predictions/test_probs_updated.csv', probs, delimiter=',')
print('Predictions saved')