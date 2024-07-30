import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from imblearn.over_sampling import SMOTE

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Sidebar navigation
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Analisis", "Testing"])

# Define identity_tokenizer function
def identity_tokenizer(text):
    return text

# Function to preprocess text for word cloud
def preprocess_text_for_wordcloud(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    stems = [stemmer.stem(word) for word in tokens]
    return ' '.join(stems)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    stems = [stemmer.stem(word) for word in tokens]
    return stems

# Function to compute TF, IDF, and TF-IDF
def compute_tfidf(docs):
    terms = list(set([term for doc in docs for term in preprocess_text(doc)]))
    tf = pd.DataFrame(index=terms)
    for i, doc in enumerate(docs):
        tf[f'D{i+1}'] = pd.Series(preprocess_text(doc)).value_counts()
    tf = tf.fillna(0).astype(int)
    tf['idf'] = np.log(len(docs) / tf.astype(bool).sum(axis=1))
    tfidf = tf.drop('idf', axis=1).multiply(tf['idf'], axis=0)
    return tf, tfidf, tf['idf']

# Page: Analisis
if menu == "Analisis":
    st.title("Halaman Analisis")
    st.write("Ini adalah halaman untuk melakukan analisis data.")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        st.subheader("Original Dataset")
        st.write(df)
        
        # Keep only the relevant columns
        cleaned_df = df[['Komentar', 'Label']]
        
        st.subheader("Cleaned Dataset")
        st.write(cleaned_df)
        
        cleaned_df['case_folded'] = cleaned_df['Komentar'].str.lower()
        
        st.subheader("Case Folded Dataset")
        st.write(cleaned_df[['case_folded', 'Label']])
        
        cleaned_df['tokenized'] = cleaned_df['case_folded'].apply(word_tokenize)
        
        st.subheader("Tokenized Dataset")
        st.write(cleaned_df[['tokenized', 'Label']])
        
        stop_words = set(stopwords.words('indonesian'))
        cleaned_df['no_stopwords'] = cleaned_df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
        
        st.subheader("Stopword Removed Dataset")
        st.write(cleaned_df[['no_stopwords', 'Label']])
        
        cleaned_df['stemmed'] = cleaned_df['no_stopwords'].apply(lambda x: [stemmer.stem(word) for word in x])
        
        st.subheader("Stemmed Dataset")
        st.write(cleaned_df[['stemmed', 'Label']])
        
        # Generate Word Cloud for Positive and Negative Labels
        positive_comments = ' '.join(cleaned_df[cleaned_df['Label'] == 'Positif']['stemmed'].apply(lambda x: ' '.join(x)))
        negative_comments = ' '.join(cleaned_df[cleaned_df['Label'] == 'Negatif']['stemmed'].apply(lambda x: ' '.join(x)))

        wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
        wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)

        st.subheader("Word Cloud for Positive Comments")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_positive, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        st.subheader("Word Cloud for Negative Comments")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_negative, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        # Compute TF, IDF, and TF-IDF
        docs = cleaned_df['Komentar'].tolist()
        tf, tfidf, idf = compute_tfidf(docs)
        
        st.subheader("Term Frequency (TF)")
        st.write(tf)
        
        st.subheader("Inverse Document Frequency (IDF)")
        st.write(idf)
        
        st.subheader("TF-IDF")
        st.write(tfidf)

        # Initialize and fit TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
        X_tfidf = vectorizer.fit_transform(cleaned_df['stemmed'].apply(lambda x: ' '.join(x)))

        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        X = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        y = cleaned_df['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Display the number of data points in training and testing sets
        st.subheader("Number of Data Points")
        st.write(f"Training Data: {len(X_train)}")
        st.write(f"Testing Data: {len(X_test)}")
        st.write(f"Training Data after SMOTE: {len(X_train_smote)}")
        
        # Pie chart for label distribution
        label_counts = cleaned_df['Label'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Label Distribution")
        st.pyplot(fig)

        # Counts of positive and negative labels in training and testing sets
        train_label_counts = y_train.value_counts()
        train_label_counts_smote = pd.Series(y_train_smote).value_counts()
        test_label_counts = y_test.value_counts()

        st.subheader("Label Distribution in Training and Testing Sets")
        st.write("Training Set")
        st.write(train_label_counts)
        st.write("Training Set after SMOTE")
        st.write(train_label_counts_smote)

        st.write("Testing Set")
        st.write(test_label_counts)

        if st.button("Train Model"):
            # Train KNeighborsClassifier with k=5
            k = 5
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train_smote, y_train_smote)

            with open('knn_model.pkl', 'wb') as f:
                pickle.dump(model, f)

            st.success(f"Model trained with k={k} and saved as knn_model.pkl")
        
        if st.button("Test Model"):
            with open('knn_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            y_pred = model.predict(X_test)

            # Display actual and predicted labels
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.subheader("Actual vs Predicted Labels")
            st.write(results_df)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'])
            cm = confusion_matrix(y_test, y_pred)
            
            st.subheader("Model Evaluation")
            st.write(f"Accuracy: {accuracy:.2f}")
            
            st.text("Classification Report:")
            st.text(report)
            
            st.text("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

elif menu == "Testing":
    st.title("Halaman Testing")
    st.write("Ini adalah halaman untuk melakukan testing model.")

    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)

    option = st.selectbox("Choose an option", ["Upload a file", "Enter a comment manually"])

    if option == "Upload a file":
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.subheader("Original Dataset")
            st.write(df)

            # Preprocess the comments
            df['case_folded'] = df['Komentar'].str.lower()
            df['tokenized'] = df['case_folded'].apply(word_tokenize)
            df['no_stopwords'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
            df['stemmed'] = df['no_stopwords'].apply(lambda x: [stemmer.stem(word) for word in x])

            st.subheader("Processed Dataset")
            st.write(df[['Komentar', 'stemmed']])

            # Vectorize and predict
            X_test = vectorizer.transform(df['stemmed'].apply(lambda x: ' '.join(x)))
            y_pred = model.predict(X_test)

            # Display results
            df['Predicted Label'] = y_pred
            st.subheader("Predicted Labels")
            st.write(df[['Komentar', 'Predicted Label']])

    elif option == "Enter a comment manually":
        comment = st.text_area("Enter your comment", "")
        if st.button("Predict"):
            stemmed_comment = preprocess_text(comment)
            X_comment = vectorizer.transform([' '.join(stemmed_comment)])
            prediction = model.predict(X_comment)
            st.subheader("Prediction")
            st.write(f"Predicted Label: {prediction[0]}")
