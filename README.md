# Wikipedia Search Engine

This is a straightforward search engine designed to search for Wikipedia articles in both English and Indonesian languages. The search engine is implemented using Python and Flask. The search engine allows users to input a query, and it returns the top 3 most relevant documents based on cosine similarity scores.

## Features

- Tokenizes and normalizes documents using nltk and Sastrawi libraries
- Implements the Bag-of-Words model for document representation
- Computes cosine similarity between the query and documents
- Ranks and displays the top 3 most similar documents

## Requirements

- Python 3.x
- Flask
- nltk
- pandas
- numpy
- Sastrawi

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/jiienc/WikiRecommender.git

## Usage
1. Place your document files in the input_directory specified in main.py
2. Run the Flask application:
   
   ```shell
   git clone https://github.com/jiienc/WikiRecommender.git

3. Open your web browser and navigate to [http://localhost:5000](http://localhost:5000).
4. Enter a query in the search box and click the "Search" button.
5. The top 3 most relevant documents will be displayed, along with their similarity scores.

## Customization

- To change the input directory for document files, modify the input_directory_path variable in the normalize_articles function in main.py
- To change the supported languages and their respective stop words, modify the stop_words and stemmer variables in the normalize_articles function.
