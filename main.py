import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_dataset_b2w():
    df = pd.read_csv('B2W-Reviews01.csv')
    df = df.dropna(subset=['review_text'])
    df['sentimento'] = df['overall_rating'].apply(lambda x: 1 if x >= 4 else 0)
    textos = df['review_text'].tolist()
    labels = df['sentimento'].tolist()
    return textos, labels

def read(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as f:
        texto = f.read()
    return pd.DataFrame({'texto': [texto]})

vectorizer = TfidfVectorizer()
model = LogisticRegression()

def treinar_modelo(textos, labels):
    textos_train, textos_test, y_train, y_test = train_test_split(textos, labels, test_size=0.2, random_state=42)
    X_train = vectorizer.fit_transform(textos_train)
    X_test = vectorizer.transform(textos_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Acurácia: {accuracy_score(y_test, y_pred):.2f}')
    return model

def prever_sentimento(texto):
    features = vectorizer.transform([texto])
    previsao = model.predict(features)[0]
    return "positivo" if previsao == 1 else "negativo"

if __name__ == "__main__":
    textos, labels = load_dataset_b2w()

    modelo = treinar_modelo(textos, labels)

    df_texto = read('text.txt')
    texto = df_texto['texto'][0]

    resultado = prever_sentimento(texto)
    print(resultado)
