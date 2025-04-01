import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

mbti_data = pd.read_csv("./MBTI 500.csv")
mbti_data

mbti_data.info()

print("Duplicates: ",mbti_data.duplicated().sum())

mbti_data.isna().count()

## Tokenization and Lemmatization
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
    text = re.sub(r'[\(\){}<>:;]', '', text) # Remove emoticons
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

mbti_data['clean_posts'] = mbti_data['posts'].apply(clean_text)
mbti_data

mbti_data[mbti_data['clean_posts']=='']

mbti_data.drop(index=22928, inplace=True)

mbti_data.to_csv('mbti_data_cleaned.csv', index=False)

# Encodeing
mbti_data['I/E'] = mbti_data['type'].apply(lambda x: 1 if x[0] == 'I' else 0)
mbti_data['S/N'] = mbti_data['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
mbti_data['F/T'] = mbti_data['type'].apply(lambda x: 1 if x[2] == 'F' else 0)
mbti_data['J/P'] = mbti_data['type'].apply(lambda x: 1 if x[3] == 'J' else 0)
mbti_data

## Text vectorization
word2vec_path = 'GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def text_to_vector(text, model, vector_size=300):
    tokens = word_tokenize(text)
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) == 0:  # If no words in the text exist in the word embedding model
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# Convert clean_posts to vector
mbti_data['w2v_vector'] = mbti_data['clean_posts'].apply(lambda x: text_to_vector(x, word2vec_model))
mbti_data

mbti_data.to_csv('mbti_data_cleaned.csv', index=False)

## SMOTE
X_smote = np.array(mbti_data['w2v_vector'].tolist())
y_smote = mbti_data['type']

# Define the desired number of samples for each class
desired_count = 1000

# Define the oversampling strategy for SMOTE
oversample_strategy = {label: desired_count for label, count in y_smote.value_counts().items() if count < desired_count}

# Define the undersampling strategy for RandomUnderSampler
undersample_strategy = {label: desired_count for label, count in y_smote.value_counts().items() if count > desired_count}

# Create the SMOTE and RandomUnderSampler objects
smote = SMOTE(sampling_strategy=oversample_strategy, random_state=42)
undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)

# Combine SMOTE and RandomUnderSampler in a pipeline
pipeline = Pipeline(steps=[
    ('smote', smote), 
    ('undersample', undersampler)
])

# Print class distribution before resampling
print("Before resampling:", y_smote.value_counts())
print()

# Apply the pipeline to resample the dataset
X_resampled, y_resampled = pipeline.fit_resample(X_smote, y_smote)

# Print class distribution after resampling
print("After resampling:", y_resampled.value_counts())

X_resampled_list = [list(row) for row in X_resampled]

# Combine X_resampled and y_resampled
resampled_df = pd.DataFrame({
    "type": y_resampled,
    "w2v_vector": X_resampled_list
})
resampled_df

# Final Encodeing
resampled_df['I/E'] = resampled_df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)
resampled_df['S/N'] = resampled_df['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
resampled_df['F/T'] = resampled_df['type'].apply(lambda x: 1 if x[2] == 'F' else 0)
resampled_df['J/P'] = resampled_df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)
resampled_df

## EDA
# Distribution for every MBTI types (Original)
plt.figure(figsize=(12, 6))
ax = sns.countplot(mbti_data['type'], order=mbti_data['type'].value_counts().index)
abs_values = mbti_data['type'].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=abs_values)
plt.title('MBTI Type Distribution')

# Save the plot to a file
plt.savefig('mbti_type_distribution.png', dpi=300, bbox_inches='tight')

plt.show()

# Calaulate the sum of eight dimentions (Original)
dimension_counts = {
    'I': mbti_data['I/E'].sum(),
    'E': len(mbti_data) - mbti_data['I/E'].sum(),
    'S': mbti_data['S/N'].sum(),
    'N': len(mbti_data) - mbti_data['S/N'].sum(),
    'F': mbti_data['F/T'].sum(),
    'T': len(mbti_data) - mbti_data['F/T'].sum(),
    'J': mbti_data['J/P'].sum(),
    'P': len(mbti_data) - mbti_data['J/P'].sum()
}

# Create dimension dataframe
dimension_df = pd.DataFrame(list(dimension_counts.items()), columns=['Dimension', 'Count'])

# Plot the distribution of eight dimentions
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Dimension', y='Count', data=dimension_df)
ax.bar_label(container=ax.containers[0], labels=dimension_df['Count'])
plt.title('Distribution of MBTI Dimensions', fontsize=16)
plt.xlabel('MBTI Dimensions', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot to a file
plt.savefig('mbti_dimensions_distribution_before_smote.png', dpi=300, bbox_inches='tight')
plt.show()

# Calaulate the sum of eight dimentions (After resampling)
dimension_counts = {
    'I': resampled_df['I/E'].sum(),
    'E': len(resampled_df) - resampled_df['I/E'].sum(),
    'S': resampled_df['S/N'].sum(),
    'N': len(resampled_df) - resampled_df['S/N'].sum(),
    'F': resampled_df['F/T'].sum(),
    'T': len(resampled_df) - resampled_df['F/T'].sum(),
    'J': resampled_df['J/P'].sum(),
    'P': len(resampled_df) - resampled_df['J/P'].sum()
}

# Create dimension dataframe
dimension_df = pd.DataFrame(list(dimension_counts.items()), columns=['Dimension', 'Count'])

# Plot the distribution of eight dimentions
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Dimension', y='Count', data=dimension_df)
ax.bar_label(container=ax.containers[0], labels=dimension_df['Count'])
plt.title('Distribution of MBTI Dimensions', fontsize=16)
plt.xlabel('MBTI Dimensions', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot to a file
plt.savefig('mbti_dimensions_distribution_after_smote.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot a word cloud for each MBTI type (Original)
def plot_wordcloud_for_mbti(df, mbti_type_column='type', post_column='clean_posts'):
    mbti_types = df[mbti_type_column].unique()
    
    plt.figure(figsize=(15, 15))
    
    for i, mbti_type in enumerate(mbti_types, 1):
        plt.subplot(4, 4, i)  # 16 types, arranged 4x4
        type_posts = df[df[mbti_type_column] == mbti_type][post_column].str.cat(sep=' ')  # Merge all posts of this type
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(type_posts)  # Generate word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(mbti_type)
    
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('mbti_type_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_wordcloud_for_mbti(mbti_data)

## Models
X = np.array(resampled_df['w2v_vector'].tolist())
y_columns = ['I/E', 'S/N', 'F/T', 'J/P']

results = {}

for y_col in y_columns:
    y = resampled_df[y_col].values
    
    # Split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # print(X_val.shape)
    # print(y_val.shape)

    # Define a function to create LSTM or GRU models
    def create_model(model_type):
        model = Sequential()
        if model_type == 'LSTM':
            model.add(Bidirectional(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))
        elif model_type == 'GRU':
            model.add(Bidirectional(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        return model

    for model_type in ['LSTM', 'GRU']:
        # Create the model
        model = create_model(model_type)
        
        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            shuffle=True,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate the model
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1_score = f1_score(y_test, y_pred)
        classification_metrics = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)
        
        # Store results
        results[f"{y_col}_{model_type}"] = {
            "training_time": training_time,
            "test_accuracy": test_accuracy,
            "test_f1_score": test_f1_score,      
            "classification_report": classification_metrics,
            "history": history.history
        }
        
        # Print evaluation results
        print(f"\nResults for Bidirectional {model_type} on {y_col}:")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1_score:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot loss and accuracy curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_type} Loss for {y_col}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_type} Accuracy for {y_col}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Store final results
print("\nFinal Results Summary:")
for key, value in results.items():
    print(f"{key}: Test Accuracy: {value['test_accuracy']:.4f}, Training Time: {value['training_time']:.2f} seconds")