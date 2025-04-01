import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('MBTI 500.csv')
print(df.info)
df=df.dropna()
print(df.info)

# Create 4 dimension
df['I/E'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)
df['S/N'] = df['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
df['F/T'] = df['type'].apply(lambda x: 1 if x[2] == 'F' else 0)
df['P/J'] = df['type'].apply(lambda x: 1 if x[3] == 'P' else 0)

print(df)

# Lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define All MBTI types
mbti_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
              'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URL
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'@\w+', '', text) # Remove users ID
    for mbti in mbti_types:
        text = re.sub(mbti, '', text, flags=re.IGNORECASE) # Remove MBTI types (both upper- and lower-case)
    text = re.sub(r'[\(\){}<>:;]', '', text) # Remove emoticons (?)
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_posts'] = df['posts'].apply(clean_text)
print(df.head())

# Define the file path
file_path = 'cleaned_data.csv'

# Export as csv file
df.to_csv(file_path, index=False, encoding='utf-8-sig')