import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

data=pd.read_csv('fake_or_real_news.csv')
print(data)

#check data
print(data.head())
print('duplicated')
print(data.duplicated().sum())
print('null')
print(data.isnull().sum())
print('unique')
print(data.nunique())
print('shape')
print(data.shape)
print('describe')
print(data.describe())
print('info')
print(data.info())

#drop not req columns
data.drop(data.columns[data.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
print(data)

#check data
print(data.head())
print('duplicated')
print(data.duplicated().sum())
print('null')
print(data.isnull().sum())
print('unique')
print(data.nunique())
print('shape')
print(data.shape)
print('describe')
print(data.describe())
print('info')
print(data.info())

#fill nulls 
data=data.fillna('')
print(data.isnull().sum())
#drop nulls
data=data.dropna()
print(data)

#lable - unique,count
print(data['label'].nunique())
print(data['label'].value_counts())

categories = pd.DataFrame({'label' : ['REAL', 'FAKE']})
print(categories)
unique_types_main = set(data['label'].unique())
inconsistent_cats = unique_types_main.difference(categories['label'])
print(inconsistent_cats)
inconsistent = data['label'].isin(inconsistent_cats)
inconsistent_rows = data[inconsistent]
print(inconsistent_rows)
data=data.drop(inconsistent_rows.index)
print('unique labels')
print(data['label'].nunique())
print(data['label'].value_counts())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['label']=le.fit_transform(data.label)
print(data.reset_index(inplace=True))
print(data)
print(data.info())



X=data.drop('label',axis=1)
y=data['label']
print(X)
print(y)
print(X.shape)
print(y.shape)
print(y.value_counts())


import string
import re
import nltk
from nltk.util import pr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
stopword = set(stopwords.words('english'))
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score,roc_auc_score
from sklearn import metrics
import itertools


ps = PorterStemmer()
corpus = []

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

data['content'] = data['title']+' '+data['text']

x = np.array(data["content"])
y = np.array(data["label"])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x)
vectorizer.get_feature_names()
print(X.shape)

first_vector = X[0]
dataframe = pd.DataFrame(first_vector.T.todense(),
                        index = vectorizer.get_feature_names(),
                        columns = ["tfidf"])
dataframe.sort_values(by = ["tfidf"], ascending = False)

cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def Output(pred,label_test):
    # Model Generalizability Analysis
    accuracy = accuracy_score(label_test, pred)
    conf_matrix = confusion_matrix(label_test, pred)
    

    print('\033[1m' + 'Confusion Matrix' + '\033[0m') # printing in bold
    print(conf_matrix)

from sklearn.svm import SVC
linear_clf = SVC()

linear_clf.fit(X_train, y_train)
pred2 = linear_clf.predict(X_test)
score_PACA = metrics.accuracy_score(y_test, pred2)
print("accuracy:   %0.3f" % score_PACA)
cm = metrics.confusion_matrix(y_test, pred2)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])


Scores2 = Output(pred2,y_test)
print(Scores2)


print("Training Accuracy:", linear_clf.score(X_train, y_train))
print("Testing Accuracy:", linear_clf.score(X_test, y_test))


def FND():
    input_title = input('\033[1m' + '\nEnter Title:' + '\033[0m')
    input_text = input('\033[1m' + '\nEnter Text:' + '\033[0m')
    
    data = {"title": [input_title], 
        "text":[input_text]}
    
    df = pd.DataFrame(data)
    
    df.reset_index(inplace=True)

    ps = PorterStemmer()
    corpus = []

    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df['title'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    print('\n')

    X = cv.transform(corpus).toarray()
    #print(X.shape)
    
    #output = model.predict(X)
    output = linear_clf.predict(X)
    print(output)
    
    if output == 1:
         print('\033[1m' + '\nReal News Detected' + '\033[0m')
    
    else:
        print('\033[1m' + '\nFake News Detected' + '\033[0m')


print(FND())








