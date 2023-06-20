import os
import unicodedata
import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

def normalize_articles(input_directory_path, output_file_path, language):
    stop_words = set(stopwords.words(language))
    stemmer = StemmerFactory().create_stemmer() if language == 'indonesian' else None

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for filename in os.listdir(input_directory_path):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(input_directory_path, filename)
                with open(input_file_path, 'r', encoding='utf-8') as input_file:
                    text = input_file.read()
                    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
                    normalized_text = re.sub(r'\d+|[^\w\s]|_', '', normalized_text)
                    normalized_text = normalized_text.lower()
                    words = nltk.word_tokenize(normalized_text)
                    words_filtered = [word for word in words if not any(char.isdigit() for char in word) and word.lower() not in stop_words]
                    if stemmer:
                        stemmed_words = [stemmer.stem(word) for word in words_filtered]
                        normalized_text = ' '.join(stemmed_words)
                    else:
                        normalized_text = ' '.join(words_filtered)
                    output_file.write(normalized_text + '\n')

# normalize_articles(r'wikipedia\artikel\bahasainggris', 'normalized.txt', 'english')
# normalize_articles(r'wikipedia\artikel\bahasaindonesia', 'normalized.txt', 'indonesian')

def build_index(input_file_path):
    documents = []
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            documents.append(line.strip())

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(documents)

    index = list(vectorizer.vocabulary_.keys())
    posting_list = count_matrix.toarray().transpose()

    return vectorizer, index, posting_list

vectorizer, index, posting_list = build_index('normalized.txt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        query_list = [query]
        query_vector = vectorizer.transform(query_list)
        similarity_scores = np.squeeze(cosine_similarity(query_vector, posting_list.transpose()))
        documents_ranked = list(zip(range(len(similarity_scores)), similarity_scores))
        documents_ranked = sorted(documents_ranked, key=lambda x: x[1], reverse=True)
        top_3_documents = documents_ranked[:3]

        results = []
        for doc_index, similarity_score in top_3_documents:
            input_directory = 'wikipedia/artikel/bahasainggris' if doc_index == 0 else 'wikipedia/artikel/bahasaindonesia'
            filename = [filename for filename in os.listdir(input_directory) if filename.endswith('.txt')][doc_index]
            results.append({'filename': filename, 'similarity_score': similarity_score})

        return render_template('results.html', query=query, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()