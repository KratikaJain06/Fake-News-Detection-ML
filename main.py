#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Load dataset
df = pd.read_excel("fake_news_detection.xlsx")

#Fix column names
df.columns = df.columns.str.strip().str.lower()
print("Columns:", df.columns)

#Remove junk column if present
if 'unnamed: 4' in df.columns:
    df.drop(columns=['unnamed: 4'], inplace=True)

#Initial label check
print("\nInitial label count:\n", df["label"].value_counts())

#Fix labels FIRST (before deleting anything)
df["label"] = df["label"].astype(str).str.lower().str.strip()

df["label"] = df["label"].replace({
    "fkae": "fake",
    "faek": "fake",
    "rael": "real"
})

print("\nAfter label correction:\n", df["label"].value_counts())

#Keep only valid labels (DON'T convert to upper yet)
df = df[df["label"].isin(["fake", "real"])]

print("\nAfter filtering valid labels:\n", df["label"].value_counts())

#Clean date (but DON'T aggressively delete)
df['date'] = pd.to_datetime(df['date'], errors='coerce')

#remove only extremely bad dates
df = df[(df['date'].isna()) | (df['date'].dt.year <= 2025)]

print("\nAfter date handling:\n", df["label"].value_counts())

#Clean text (SAFE cleaning)

# Remove weird encoding
df['text'] = df['text'].astype(str).str.replace(r'[^\x00-\x7F]+', '', regex=True)
df['title'] = df['title'].astype(str).str.replace(r'[^\x00-\x7F]+', '', regex=True)

# Remove punctuation (fix regex warning properly)
df['text'] = df['text'].str.replace('[^a-zA-Z\\s]', '', regex=True)
df['title'] = df['title'].str.replace('[^a-zA-Z\\s]', '', regex=True)

# Convert to lowercase
df['text'] = df['text'].str.lower()
df['title'] = df['title'].str.lower()

# Remove ONLY completely empty text
df = df[df['text'].str.strip() != ""]

print("\nAfter text cleaning:\n", df["label"].value_counts())

#Remove duplicates
df.drop_duplicates(inplace=True)

#FINAL label format
df["label"] = df["label"].str.upper()

# Final check
print("\nFinal Dataset Info:")
print(df.head(295))
print("Shape:", df.shape)
print("Final label count:\n", df["label"].value_counts())

#Save cleaned file
df.to_csv("cleaned_fake_news.csv", index=False)

#plot graphs
counts = df["label"].value_counts()

counts.plot(kind='bar')

plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Fake vs Real News Distribution")

plt.show()

df['text_length'] = df['text'].fillna("").str.split().str.len()
fake = df[df['label'] == 'FAKE']
real = df[df['label'] == 'REAL']
fake['text_length'].plot(kind='hist', alpha=0.5, label='FAKE')
real['text_length'].plot(kind='hist', alpha=0.5, label='REAL')

plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.title("Text Length Distribution (Fake vs Real)")
plt.legend()

plt.show()

fake_length = fake['text'].fillna("").astype(str).tolist()
real_length = real['text'].fillna("").astype(str).tolist()
fake_text = " ".join(fake_length)
real_text = " ".join(real_length)
fake_words=fake_text.split()
real_words=real_text.split()
count_fake=Counter(fake_words)
count_real=Counter(real_words)
top_fake=count_fake.most_common(10)
top_real=count_real.most_common(10)
words_fake=[word for word,count in top_fake]
counts_fake=[count for word,count in top_fake]
plt.bar(words_fake,counts_fake)
plt.xticks(rotation=45)
plt.title("Top Words in Fake News")
plt.show()
words_real=[word for word,count in top_real]
counts_real=[count for word,count in top_real]
plt.bar(words_real,counts_real)
plt.xticks(rotation=45)
plt.title("Top Words in Real News")
plt.show()

#machine learning part
#model building
#tf-idf,train test split
df['text']=df['text'].fillna("").astype(str)
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=200)

vectorizer=TfidfVectorizer(stop_words='english')
X_train_tfidf=vectorizer.fit_transform(X_train)
X_test_tfidf=vectorizer.transform(X_test)

#model training
naive_model=MultinomialNB()
model=naive_model.fit(X_train_tfidf,y_train)
#prediction
y_pred=model.predict(X_test_tfidf)
#accuracy
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

#classification report
print(classification_report(y_test,y_pred))

#confusion matrix
matrix=confusion_matrix(y_test,y_pred)
print(matrix)