import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class Predicciones_cambio_climatico:
    def __init__(self, name_df):
        mensajesTwitter = pd.read_csv("datas/calentamientoClimatico.csv", delimiter=";")

    def normalizacion(self):
        mensaje = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', mensaje)
        mensaje = re.sub('@[^\s]+','USER', mensaje)
        mensaje = mensaje.lower().replace("ё", "е")
        mensaje = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', mensaje)
        mensaje = re.sub(' +',' ', mensaje)
        return mensaje.strip()

    def preparacion(self):
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.mensajesTwitter['CREENCIA'] = (self.mensajesTwitter['CREENCIA']=='Yes').astype(int)
        self.mensajesTwitter["TWEET"] = self.mensajesTwitter["TWEET"].apply(self.normalizacion)

        stopWords = stopwords.words('english')
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([palabra for palabra in mensaje.split() if palabra not in (stopWords)]))

        stemmer = SnowballStemmer('english')
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([stemmer.stem(palabra) for palabra in mensaje.split(' ')]))

        lemmatizer = WordNetLemmatizer()
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([lemmatizer.lemmatize(palabra) for palabra in mensaje.split(' ')]))

    def aprendizaje(self):
        X_train, X_test, y_train, y_test = train_test_split(self.mensajesTwitter['TWEET'].values,  self.mensajesTwitter['CREENCIA'].values,test_size=0.2)
        etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),('tfidf', TfidfTransformer()),('algoritmo', MultinomialNB())])
        modelo = etapas_aprendizaje.fit(X_train,y_train)








    def __str__(self):
        return f'Nuestro dataset:\n{self.mensajesTwitter.head(10)}\n Tamaño: {self.mensajesTwitter.shape[0]}'

