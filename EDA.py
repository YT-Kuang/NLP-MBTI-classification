import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_data.csv', encoding='ISO-8859-1')
print(df.head())

# 1. EDA
# 1.1. Split the review into words
df['word_count'] = df['clean_posts'].str.split().str.len()

# 1.2. Draw the plot
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=30, color='red', edgecolor='black')
plt.title('Distribution of Word Counts in Reviews')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
df = df.drop(['word_count'], axis=1)