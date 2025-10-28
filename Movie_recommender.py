#Movie recommender: Amit Kriaf

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load and Prepare Data 
def load_data_and_build_models():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    
    # Collaborative Filtering model (SVD)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42) 
    model = SVD()
    model.fit(trainset)

    # Content-Based model
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies, ratings, model, cosine_sim


# Recommendation Functions 

#this function tells recommendations based on other people's ratings (using SVD model which learns from previous data).
def get_cf_recommendations(user_id, movies, ratings, model, top_rec=5):
    all_movie_ids = movies['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    movies_to_predict = [m for m in all_movie_ids if m not in rated_movies]

    predictions = [(mid, model.predict(user_id, mid).est) for mid in movies_to_predict]
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = predictions[:top_rec]
    return [movies.loc[movies['movieId'] == mid, 'title'].values[0] for mid, _ in top_movies]

#this function tells recommendations based on proximity to the genre of a given movie
def get_similar_movies(title, movies, cosine_sim, top_rec=5):
    title_corrected = find_movie_title(title, movies)
    if not title_corrected:
        print(f"Movie '{title}' not found in the database.")
        return []
    
    idx = movies[movies['title'] == title_corrected].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_rec+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

#this function tells recommendations with an hybrid approch, using alpha score to decide the ratio between the two models.
def get_hybrid_recommendations(user_id, liked_movie, movies, ratings, model, cosine_sim, top_rec=5, alpha=0.5):
    cf_scores = {}
    cb_scores = {}

    # Collaborative Filtering (CF) part 
    if user_id is not None:
        cf_list = get_cf_recommendations(user_id, movies, ratings, model, top_rec*2)
        for title in cf_list:
            mid = movies.loc[movies['title'] == title, 'movieId'].values[0]
            cf_scores[mid] = model.predict(user_id, mid).est

    # Content-Based part 
    if liked_movie is not None:
        title_corrected = find_movie_title(liked_movie, movies)
        if not title_corrected:
            print(f" Movie '{liked_movie}' not found. Skipping Content-Based part.")
        else:
            cb_list = get_similar_movies(title_corrected, movies, cosine_sim, top_rec*2)
            liked_idx = movies[movies['title'] == title_corrected].index[0]
            for title in cb_list:
                mid = movies.loc[movies['title'] == title, 'movieId'].values[0]
                idx = movies[movies['title'] == title].index[0]
                cb_scores[mid] = cosine_sim[idx, liked_idx]

    # Combine scores 
    hybrid_scores = {}
    for mid in set(cf_scores.keys()) | set(cb_scores.keys()):
        cf = cf_scores.get(mid, 0)
        cb = cb_scores.get(mid, 0)
        hybrid_scores[mid] = alpha * cf + (1 - alpha) * cb

    top_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_rec]
    return [movies.loc[movies['movieId'] == mid, 'title'].values[0] for mid, _ in top_movies]

def find_movie_title(title, movies):
    #checks if there's a movie in the database that is similar to the title the user entered
    titles = movies['title'].tolist()
    match = get_close_matches(title, titles, n=1, cutoff=0.6)
    return match[0] if match else None

# ---------- Input & Main Loop ----------
def get_input_or_exit(prompt):
    user_input = input(prompt).strip()
    if user_input.lower() == 'exit':
        print("Exiting...")
        exit()
    return user_input


def main():
    print("🎥 Welcome to the Movie Recommender!")
    print("Type 'exit' anytime to quit.\n")

    movies, ratings, model, cosine_sim = load_data_and_build_models()

    while True:
        user_id_input = get_input_or_exit("Enter your User ID (or press Enter to skip): ")
        user_id = int(user_id_input) if user_id_input else None

        liked_movie = get_input_or_exit("Enter a movie you liked (or press Enter to skip): ")
        liked_movie = liked_movie if liked_movie else None

        top_rec_input = get_input_or_exit("How many recommendations do you want? (default 5): ")
        top_rec = int(top_rec_input) if top_rec_input else 5

        alpha_input = get_input_or_exit("Alpha (weight for CF in Hybrid, default 0.5): ")
        alpha = float(alpha_input) if alpha_input else 0.5

        print("\nChoose recommendation type:")
        print("1 - Collaborative Filtering")
        print("2 - Content-Based")
        print("3 - Hybrid")
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "1" and user_id is not None:
            recs = get_cf_recommendations(user_id, movies, ratings, model, top_rec)
            print(f"\nTop {top_rec} CF recommendations for user {user_id}:\n")

        elif choice == "2" and liked_movie is not None:
            recs = get_similar_movies(liked_movie, movies, cosine_sim, top_rec)
            print(f"\nTop {top_rec} movies similar to '{liked_movie}':\n")

        elif choice == "3" and user_id is not None and liked_movie is not None:
            recs = get_hybrid_recommendations(user_id, liked_movie, movies, ratings, model, cosine_sim, top_rec, alpha)
            print(f"\nTop {top_rec} hybrid recommendations for user {user_id} based on '{liked_movie}':\n")

        else:
            print("Missing information or invalid choice. Try again.\n")
            continue

        for i, movie in enumerate(recs, 1):
            print(f"{i}. {movie}")

        again = input("\nWould you like another recommendation? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            print("\n🍿 Enjoy your movies! Goodbye!")
            break


# ---------- Run Program ----------
if __name__ == "__main__":
    main()
