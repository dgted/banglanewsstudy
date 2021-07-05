from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.7)
X = tfidfconverter.fit_transform(all_text).toarray()