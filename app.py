from flask import Flask, flash, request, redirect, url_for, render_template
import joblib
import spacy
import gensim.downloader as api


wv = api.load('word2vec-google-news-300')
nlp = spacy.load("en_core_web_lg") # if this fails then run "python -m spacy download en_core_web_lg" to download that model


def preprocess_and_vectorize(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
        
    return wv.get_mean_vector(filtered_tokens)



app = Flask(__name__)
model = joblib.load('fake_news_classification.pkl')

@app.route('/')
def my_form():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    return text
def show_result():
    test_news_vectors = [preprocess_and_vectorize(n) for n in my_form_post()]
    result = model.predict(test_news_vectors)
    flash(result)
    return render_template('index.html')


if __name__ == "__main__":
    app.run()