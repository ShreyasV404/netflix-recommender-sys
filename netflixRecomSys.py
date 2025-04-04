import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix Recommender", layout="wide", page_icon="🎬")


# Load and preprocess dataset
@st.cache_data
def load_data():
    netflix_data = pd.read_csv("netflix_titles.csv")
    netflix_poster_data = pd.read_csv("netflix-rotten-tomatoes-metacritic-imdb.csv")

    merged_data = pd.merge(
        netflix_data, 
        netflix_poster_data, 
        left_on='title', 
        right_on='Title', 
        how='inner'  
    )

    merged_data.drop(columns=['Title'], inplace=True)
    threshold = 480
    merged_data = merged_data.dropna(thresh=len(merged_data) - threshold, axis=1)
    merged_data = merged_data.dropna()

    # Fix: Create combined_features column
    def combine_features(row):
        return f"{row['type']} {row['Genre']} {row['Summary']} {row['country']} {row['rating']}"

    merged_data["combined_features"] = merged_data.apply(combine_features, axis=1)

    # Reset index after processing
    merged_data.reset_index(drop=True, inplace=True)

    return merged_data

@st.cache_data
def preprocess_data(merged_data):
    df = merged_data[['release_year', 'rating', 'Runtime', 'Genre', 'Tags', 'listed_in', 'country']].dropna()

    runtime_mapping = {
        "< 30 minutes": 15, 
        "30-60 mins": 45, 
        "1-2 hour": 90, 
        "> 2 hrs": 150
    }
    df["Runtime"] = df["Runtime"].map(runtime_mapping)

    df = df.reset_index(drop=True)  
    merged_data = merged_data.loc[df.index].reset_index(drop=True)  

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[['rating', 'listed_in', 'country']])

    tfidf = TfidfVectorizer(stop_words='english')
    genre_tfidf = tfidf.fit_transform(df['Genre'])
    tags_tfidf = tfidf.fit_transform(df['Tags'])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['release_year', 'Runtime']])

    X = np.hstack((scaled_features, encoded_features, genre_tfidf.toarray(), tags_tfidf.toarray()))
    
    return X, df, merged_data

@st.cache_resource
def compute_kmeans(X):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    return clusters

@st.cache_resource
def compute_dbscan(X):
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    return dbscan.fit_predict(X)

@st.cache_resource
def compute_hierarchical(X):
    hierarchical = AgglomerativeClustering(n_clusters=3)
    return hierarchical.fit_predict(X)

@st.cache_resource
def compute_cosine_similarity(merged_data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(merged_data["combined_features"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, model="Cosine Similarity", num_recommendations=5, merged_data=None, X=None, clusters=None, cosine_sim=None):
    if title not in merged_data["title"].values:
        return ["Movie/Show not found! Please try another title."], []

    merged_data = merged_data.reset_index(drop=True)

    idx = merged_data[merged_data["title"] == title].index[0]

    if model == "Cosine Similarity":
        similarity_scores = list(enumerate(cosine_sim[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    elif model in ["KMeans", "DBSCAN", "Hierarchical"]:
        # Map model_choice to corresponding cluster column
        cluster_column = model if model != "KMeans" else "Cluster"
        cluster_label = merged_data.iloc[idx][cluster_column]
        cluster_movies = merged_data[merged_data[cluster_column] == cluster_label].index.tolist()
        cluster_movies = [i for i in cluster_movies if i < len(X)]
        similarity_scores = [(i, np.linalg.norm(X[idx] - X[i])) for i in cluster_movies if i != idx]
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1])[:num_recommendations]

    recommended_titles = [merged_data.iloc[i[0]]["title"] for i in similarity_scores]
    recommended_posters = [merged_data.iloc[i[0]]["Image"] for i in similarity_scores]
    return recommended_titles, recommended_posters


st.title("🎬 Netflix Recommender System")
st.write("Find similar movies/shows based on your favorite titles!")

merged_data = load_data()

netflix_titles_list = merged_data["title"].dropna().unique().tolist()
user_input = st.selectbox("Search a movie/show title:", sorted(netflix_titles_list))


model_choice = st.selectbox("Choose recommendation model:", ["Cosine Similarity", "KMeans", "DBSCAN", "Hierarchical"])


X, _, _ = preprocess_data(merged_data)
cosine_sim = compute_cosine_similarity(merged_data)
clusters = compute_kmeans(X)
dbscan_clusters = compute_dbscan(X)
hierarchical_clusters = compute_hierarchical(X)

merged_data["Cluster"] = clusters
merged_data["DBSCAN"] = dbscan_clusters
merged_data["Hierarchical"] = hierarchical_clusters

if st.button("Recommend"):
    if user_input in merged_data["title"].values:
        movie_data = merged_data[merged_data["title"] == user_input].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if pd.notna(movie_data['Image']) and str(movie_data['Image']).startswith("http"):  
                st.image(movie_data['Image'], width=200)
            else:
                st.warning("No poster available.")

        with col2:
            st.subheader(f"📜 Details of '{user_input}':")
            st.write(f"Genre: {movie_data['Genre']}")
            st.write(f"Description: {movie_data['Summary']}")
            st.write(f"Country: {movie_data['country']}")
            st.write(f"Rating: {movie_data['rating']}")
            st.write(f"Release Year: {movie_data['release_year']}")
            st.write(f"Link: {movie_data['Netflix Link']}")

        recommendations, posters = recommend_movies(user_input, model=model_choice, merged_data=merged_data, X=X, clusters=clusters, cosine_sim=cosine_sim)

        if "Movie/Show not found!" in recommendations:
            st.error(recommendations[0])
        else:
            st.subheader("🔍 Recommended Titles:")
            cols = st.columns(len(recommendations))
            for i in range(len(recommendations)):
              with cols[i]:
                st.image(posters[i], width=120)
                with st.expander(f"🔽 Show details for {recommendations[i]}"):
                  movie_data = merged_data[merged_data["title"] == recommendations[i]].iloc[0]
                  st.write(f"Genre: {movie_data['Genre']}")
                  st.write(f"Description: {movie_data['Summary']}")
                  st.write(f"Country: {movie_data['country']}")
                  st.write(f"Rating: {movie_data['rating']}")
                  st.write(f"Release Year: {movie_data['release_year']}")
                  st.write(f"Link: {movie_data['Netflix Link']}")


st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    use_container_width=True
)

st.sidebar.markdown("""
Welcome to the **Netflix Recommender System**! 🍿  
Find movies and shows similar to your favorites.

---

### ✨ Features:
- 🔍 Search any Netflix title  
- 🤖 Choose from 4 recommendation models:  
  `Cosine Similarity`, `KMeans`, `DBSCAN`, or `Hierarchical`
- 🖼️ View posters and detailed info
- ⚡ Powered by machine learning


""")

st.sidebar.markdown("---")
st.sidebar.markdown("📧 *Developed by*")
st.sidebar.markdown("👨🏻‍💻 *B018 Shreyas Vikhare | B012 Tanmay Nikam*")
st.sidebar.markdown("👨🏻‍💻 *B027 Yash Gadhave    | B032 Sagar Yadav*")
st.sidebar.markdown("👨🏻‍💻 *B051 Tirtha Parmar    | B055 Rishi Shrigadi*")
st.sidebar.markdown("---")

st.sidebar.markdown("""
👨‍💻 **Built With:**
- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
                    """)
