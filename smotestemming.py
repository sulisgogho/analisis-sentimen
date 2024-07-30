import streamlit as st
import pandas as pd
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
from imblearn.over_sampling import SMOTE

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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
    return ' '.join(tokens)

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
        
        # Generate Word Cloud for Positive and Negative Labels
        positive_comments = ' '.join(cleaned_df[cleaned_df['Label'] == 'Positif']['case_folded'].apply(preprocess_text_for_wordcloud))
        negative_comments = ' '.join(cleaned_df[cleaned_df['Label'] == 'Negatif']['case_folded'].apply(preprocess_text_for_wordcloud))

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
        
        vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(cleaned_df['no_stopwords'].apply(lambda x: ' '.join(x)))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Formatting TF-IDF results similar to the provided image
        formatted_tfidf_df = cleaned_df[['Komentar']].copy()
        formatted_tfidf_df['TF-IDF Weighted Twit'] = tfidf_df.apply(lambda row: ' '.join(f'{weight:.1f}' for weight in row), axis=1)

        st.subheader("TF-IDF Weighted Dataset")
        st.write(formatted_tfidf_df)

        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        X = tfidf_df
        y = cleaned_df['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        # Apply SMOTE to balance the training data
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
        vectorizer.tokenizer = identity_tokenizer

    def preprocess_text(text):
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('indonesian'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    st.title("Data Classification")

    option = st.selectbox("Choose input method", ("Upload a file", "Enter a comment manually"))

    if option == "Upload a file":
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            df['preprocessed'] = df['Komentar'].apply(preprocess_text)
            tfidf_matrix = vectorizer.transform(df['preprocessed'])
            predictions = model.predict(tfidf_matrix)
            df['prediction'] = predictions
            
            st.subheader("Classification Results")
            st.write(df)
            
    elif option == "Enter a comment manually":
        comment = st.text_area("Enter your comment")
        if st.button("Analyze"):
            preprocessed_comment = preprocess_text(comment)
            tfidf_vector = vectorizer.transform([preprocessed_comment])
            prediction = model.predict(tfidf_vector)[0]
            
            st.subheader("Classification Result")
            st.write(f"The comment is classified as: {prediction}")
