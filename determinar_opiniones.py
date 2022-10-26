import re
import pandas as pd

# El modulo nltk esta en desuso puesto que no funciona bien del todo
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
from sklearn.model_selection import GridSearchCV
from sklearn import svm

class Predicciones_cambio_climatico:
    '''Esta clase nos va a permitir entrenar un modelo a partir de un csv el cual predice si una persona cree o no en el cambio climático a través de una frase
    '''
    def __init__(self, name_csv):
        '''Creamos el constructor a partir del cual vamos a trabajar

        Args:
            name_csv (String): Nombre del csv
        '''


        self.mensajesTwitter = pd.read_csv(name_csv, delimiter=";")

    def normalizacion(self, mensaje):
        '''Normalizamos el mensaje identificando URL y USER, utilizando caracteres reconocibles(sin acentos) y pasando el texto a minúsculas.

        Returns:
            String: Mensaje normalizado
        '''
        mensaje = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', mensaje)
        mensaje = re.sub('@[^\s]+','USER', mensaje)
        mensaje = mensaje.lower().replace("ё", "е")
        mensaje = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', mensaje)
        mensaje = re.sub(' +',' ', mensaje)
        return mensaje.strip()

    def preparacion(self):
        '''
        1 - Descargamos las bibliotecas de palabras.
        2 - Normalizamos.
        3 - Eliminamos palabras irrelevantes (Stop words).
        4 - Eliminamos prefijos y sufijos irrelevantes (Stemming).
        5 - Ajustamos las palabras con los prefijos y sufijos eliminados anteriormente (Lematización).
        '''

        # 1
        nltk.download('stopwords')
        nltk.download('wordnet')

        # 2
        self.mensajesTwitter['CREENCIA'] = (self.mensajesTwitter['CREENCIA']=='Yes').astype(int)
        self.mensajesTwitter["TWEET"] = self.mensajesTwitter["TWEET"].apply(self.normalizacion)

        # 3
        self.stopWords = stopwords.words('english')
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([palabra for palabra in mensaje.split() if palabra not in (self.stopWords)]))

        # 4
        self.stemmer = SnowballStemmer('english')
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([self.stemmer.stem(palabra) for palabra in mensaje.split(' ')]))

        # 5
        self.lemmatizer = WordNetLemmatizer()
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([self.lemmatizer.lemmatize(palabra) for palabra in mensaje.split(' ')]))

    def dividir_dataset(self):
        '''
        Separamos nuestro dataset en variables de entrenamiento y de test (con un 80% y 20% respectivamente)
        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.mensajesTwitter['TWEET'].values,  self.mensajesTwitter['CREENCIA'].values,test_size=0.2)

    def aprendizaje_bayesiano(self):
        '''Entrenamos nuestro modelo siguiendo el modelo bayesiano
        '''

        # Pipeline canaliza diferentes funciones de aprendizaje:
        #   1 - CountVectorizer: Crea una matriz con el numero de veces que aparece cada palabra
        #
        #   2 - TfidfTransformer: Expresa la escasez de la palabra en el conjunto de los mensajes
        #           Tfid: Disminuye cuando una palabra está presente en muchos mensajes,
        #                 también disminuye cuando está poco presente en un mensaje
        #                 y es máximo para las palabras poco frecuentes que aparecen mucho en el conjunto de los mensajes que tenemos para analizar.
        #
        #   3 - MultinomialNB: Uso del algoritmo bayesiano ingenuo múltiple.

        etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()), ('tfidf', TfidfTransformer()), ('algoritmo', MultinomialNB())])

        # Ajustamos el modelo a nuestras variables de entrenamiento
        # .fit() es el responsable el responsable de entrenar el modelo
        self.modelo = etapas_aprendizaje.fit(self.X_train, self.y_train)


    def buscarC_optimo(self, X_train, y_train, etapas_aprendizaje):
        '''Buscamos el numero de clusters óptimos para nuestro modelo

        Args:
            X_train (pd.Dataframe): Variables independientes del entrenamiento
            y_train (np.array): Variable dependiente de entrenamiento
            etapas_aprendizaje (Pipeline): Pasos en el aprendizaje

        Returns:
            int: Numero de clusters optimos
        '''

        parametrosC = {'algoritmo__C':(1,2,4,5,6,7,8,9,10,11,12)}
        busquedaCOptimo = GridSearchCV(etapas_aprendizaje, parametrosC, cv = 2)
        busquedaCOptimo.fit(X_train,y_train)

        return list(busquedaCOptimo.best_params_.values())[0]


    def aprendizaje_svm(self):
        '''Entrenamos nuestro modelo utilizando el modelo Support Vector Machine (algoritmo de aprendizaje supervisado)
        '''

        # Separamos nuestro dataset igual que antes
        self.dividir_dataset()

        # Canalizamos nuevamente mediante pipeline, sin embargo, ahora indicamos el nuevo aprendizaje que vamos a seguir(SVM)
        etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),('tfidf', TfidfTransformer()),('algoritmo', svm.SVC(kernel = 'linear', C = 2))])

        # Calculamos el numero optimo de clusters
        n_clusters = self.buscarC_optimo(self.X_train, self.y_train, etapas_aprendizaje)

        # Entrenamos nuevamente el modelo, solo que esta vez nos aseguramos de que tiene el numero correcto de clusters
        etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),('tfidf', TfidfTransformer()),('algoritmo', svm.SVC(kernel='linear', C = n_clusters))])
        self.modelo = etapas_aprendizaje.fit(self.X_train, self.y_train)




    def predecir_frase(self, frase):
        '''Esta funcion nos va a permitir predecir a partir de una frase si una persona cree en el cambio climatico o no

        Args:
            frase (String): Frase que queremos predecir

        Returns:
            Bool: Devuelve True si cree en el cambio climatico y False el caso contrario
        '''
        #Normalización
        frase = self.normalizacion(frase)

        #Eliminación de las stops words
        frase = ' '.join([palabra for palabra in frase.split() if palabra not in (self.stopWords)])

        #Aplicación de stemming
        frase =  ' '.join([self.stemmer.stem(palabra) for palabra in frase.split(' ')])

        #Lematización
        frase = ' '.join([self.lemmatizer.lemmatize(palabra) for palabra in frase.split(' ')])

        prediccion = self.modelo.predict([frase])

        if(prediccion[0]==0):
            return True

        else:
            return False

    def __str__(self):
        '''Nos devuelve el print de nuestra clase

        Returns:
            String: Información sobre la cantidad de observaciones y su contenido
        '''

        # Classification_report: Printeamos un grafico que nos muestra algunas características de nuestro modelo
        #         Observamos la precisión y la validez del mismo mediante la función predict y nuestras variables de testeo
        return f'Nuestro dataset:\n{self.mensajesTwitter.head(10)}\n\nTamaño: {self.mensajesTwitter.shape[0]}\n\nNuestro reporte de clasificacion del modelo:\n{classification_report(self.y_test, self.modelo.predict(self.X_test), digits=4)}'


