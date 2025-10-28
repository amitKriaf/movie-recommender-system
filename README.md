# Movie Recommender System 

This project allows users to get movie recommendations based on:  
- Other users' ratings (Collaborative Filtering)  
- Similarity to movies they liked (Content-Based)  
- A combination of both approaches (Hybrid)
- 
## Features

- **Collaborative Filtering**: Predicts ratings for unseen movies using SVD from the `surprise` library.  
- **Content-Based Filtering**: Recommends movies with similar genres using TF-IDF and cosine similarity.  
- **Hybrid Recommendations**: Combines CF and CB scores with adjustable weight (`alpha`).  
- Handles **misspelled or partial movie names** using fuzzy matching.
