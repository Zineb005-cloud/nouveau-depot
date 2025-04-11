from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Charger des données de texte
data = fetch_20newsgroups(subset='train', categories=['rec.sport.hockey', 'sci.space'])

# Créer des labels simples (0 = sport, 1 = espace)
X = data.data
y = [0 if label == 15 else 1 for label in data.target]  # adapter si catégories changent

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline de transformation + modèle
vect = CountVectorizer()
clf = MultinomialNB()

model = Pipeline([('vect', vect), ('clf', clf)])
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, "model_sentiment.pkl")
