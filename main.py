import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import yfinance as yf
from transformers import BertTokenizer, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_earnings_call_transcript(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator=' ')
    return text

url = "https://www.fool.com/earnings/call-transcripts/2025/03/21/fedex-fdx-q3-2025-earnings-call-transcript/"
transcript = get_earnings_call_transcript(url)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^a-zA-Z\s.,?!]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

clean_text = preprocess(transcript)
sentences = nltk.sent_tokenize(clean_text)

def analyze_sentiment(sentences, max_length=256):
    sentiment_scores = []
    chunk = ""
    current_length = 0

    for sentence in sentences:
        tokenized_length = len(tokenizer.tokenize(sentence))

        if current_length + tokenized_length > max_length and chunk:
            result = sentiment_analyzer(chunk[:max_length])
            for res in result:
                sentiment_scores.append((chunk, res['label'], res['score']))
            chunk = ""
            current_length = 0

        chunk += " " + sentence
        current_length += tokenized_length

    if chunk: 
        result = sentiment_analyzer(chunk[:max_length])
        for res in result:
            sentiment_scores.append((chunk, res['label'], res['score']))

    return pd.DataFrame(sentiment_scores, columns=["Sentence", "Sentiment", "Score"])

sentiment_df = analyze_sentiment(sentences)

sentiment_summary = sentiment_df.groupby("Sentiment").size().reset_index(name='Count')
print("Sentiment Summary:")
print(sentiment_summary)

print("\nSample Sentiment Results:")
for i, row in sentiment_df.head(10).iterrows():
    print(f"Sentence: {row['Sentence'][:200]}...")  
    print(f"Sentiment: {row['Sentiment']} | Score: {row['Score']}\n")

stock_ticker = 'FDX'
earnings_date = datetime.date(2025, 3, 20) # Earnings date from the transcript

stock_data = yf.download(stock_ticker, start=earnings_date - pd.Timedelta(days=7),
                         end=earnings_date + pd.Timedelta(days=7))
stock_data['Price_Change'] = stock_data['Close'].pct_change()

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

sns.barplot(x='Sentiment', y='Count', data=sentiment_summary, ax=axes[0], palette='viridis')
axes[0].set_title("Distribution of Sentiments from Earnings Call Transcript")

axes[1].plot(stock_data.index, stock_data['Close'], marker='o', color='b', label='Close Price')
axes[1].plot(stock_data.index, stock_data['Price_Change'], marker='x', color='r', label='Price Change')
axes[1].axvline(earnings_date, color='g', linestyle='--', label='Earnings Date')
axes[1].set_title(f"{stock_ticker} Stock Price Around Earnings Date")
axes[1].legend()

plt.tight_layout()
plt.show()