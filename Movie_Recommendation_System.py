import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import random
from datetime import datetime
from collections import defaultdict, Counter
import re

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class AdvancedMovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.knn_model = None
        self.svd_model = None
        self.user_item_matrix = None
        self.movie_similarity_matrix = None
        
    def generate_comprehensive_data(self):
        """Generate comprehensive sample data with more features"""
        # Enhanced movies data
        movies_data = {
            'movieId': list(range(1, 51)),
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Star Wars: A New Hope',
                'The Lord of the Rings: The Fellowship of the Ring',
                'Fight Club', 'The Godfather: Part II', 'The Empire Strikes Back',
                'Return of the King', 'Schindler\'s List', 'The Prestige',
                'Interstellar', 'The Departed', 'The Green Mile', 'Gladiator',
                'The Lion King', 'Avengers: Infinity War', 'Parasite', 'Joker',
                'Whiplash', 'The Social Network', 'La La Land', 'Get Out',
                'Black Panther', 'Django Unchained', 'The Wolf of Wall Street',
                'Gone Girl', 'Mad Max: Fury Road', 'The Revenant', 'Arrival',
                'Birdman', 'Spotlight', 'The Grand Budapest Hotel', 'Her',
                'Nightcrawler', 'Baby Driver', 'The Martian', 'Gravity',
                'Blade Runner 2049', 'Drive', 'John Wick', 'Mission: Impossible',
                'The Incredibles', 'Spirited Away'
            ],
            'genres': [
                'Drama', 'Crime|Drama', 'Action|Crime|Drama', 'Crime|Drama',
                'Drama|Romance', 'Action|Sci-Fi|Thriller', 'Action|Sci-Fi',
                'Biography|Crime|Drama', 'Crime|Thriller|Horror', 'Action|Adventure|Fantasy',
                'Adventure|Fantasy|Drama', 'Drama', 'Crime|Drama', 'Action|Adventure|Fantasy',
                'Adventure|Fantasy|Drama', 'Biography|Drama|History', 'Drama|Mystery|Thriller',
                'Adventure|Drama|Sci-Fi', 'Crime|Drama|Thriller', 'Crime|Drama|Fantasy',
                'Action|Adventure|Drama', 'Animation|Adventure|Drama', 'Action|Adventure|Sci-Fi',
                'Comedy|Drama|Thriller', 'Crime|Drama|Thriller', 'Drama|Music',
                'Biography|Drama', 'Comedy|Drama|Music', 'Horror|Mystery|Thriller',
                'Action|Adventure|Sci-Fi', 'Drama|Western', 'Biography|Comedy|Crime',
                'Drama|Mystery|Thriller', 'Action|Adventure|Sci-Fi', 'Action|Adventure|Drama',
                'Drama|Mystery|Sci-Fi', 'Comedy|Drama', 'Biography|Crime|Drama',
                'Adventure|Comedy|Drama', 'Drama|Romance|Sci-Fi', 'Crime|Drama|Thriller',
                'Action|Crime|Drama', 'Adventure|Drama|Sci-Fi', 'Drama|Sci-Fi|Thriller',
                'Drama|Mystery|Sci-Fi', 'Crime|Drama', 'Action|Crime|Thriller',
                'Action|Adventure|Thriller', 'Animation|Action|Adventure', 'Animation|Adventure|Family'
            ],
            'year': [
                1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 1991, 1977,
                2001, 1999, 1974, 1980, 2003, 1993, 2006, 2014, 2006, 1999,
                2000, 1994, 2018, 2019, 2019, 2014, 2010, 2016, 2017, 2018,
                2012, 2013, 2014, 2015, 2015, 2016, 2014, 2015, 2014, 2013,
                2014, 2017, 2015, 2013, 2017, 2011, 2014, 2018, 2004, 2001
            ],
            'director': [
                'Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan', 'Quentin Tarantino',
                'Robert Zemeckis', 'Christopher Nolan', 'Lana Wachowski', 'Martin Scorsese',
                'Jonathan Demme', 'George Lucas', 'Peter Jackson', 'David Fincher',
                'Francis Ford Coppola', 'Irvin Kershner', 'Peter Jackson', 'Steven Spielberg',
                'Christopher Nolan', 'Christopher Nolan', 'Martin Scorsese', 'Frank Darabont',
                'Ridley Scott', 'Roger Allers', 'Anthony Russo', 'Bong Joon Ho', 'Todd Phillips',
                'Damien Chazelle', 'David Fincher', 'Damien Chazelle', 'Jordan Peele',
                'Ryan Coogler', 'Quentin Tarantino', 'Martin Scorsese', 'David Fincher',
                'George Miller', 'Alejandro G. Iñárritu', 'Denis Villeneuve', 'Alejandro G. Iñárritu',
                'Tom McCarthy', 'Wes Anderson', 'Spike Jonze', 'Dan Gilroy', 'Edgar Wright',
                'Ridley Scott', 'Alfonso Cuarón', 'Denis Villeneuve', 'Nicolas Winding Refn',
                'Chad Stahelski', 'Christopher McQuarrie', 'Brad Bird', 'Hayao Miyazaki'
            ],
            'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6, 
                      8.8, 8.8, 9.0, 8.7, 8.9, 8.9, 8.5, 8.6, 8.5, 8.6,
                      8.5, 8.5, 8.4, 8.6, 8.4, 8.5, 7.7, 8.0, 7.7, 7.3,
                      8.4, 8.2, 8.1, 8.1, 8.0, 7.9, 7.7, 8.1, 8.1, 8.0,
                      7.8, 7.6, 8.0, 7.7, 8.0, 7.8, 7.4, 7.7, 8.0, 8.6],
            'votes': [2700000, 1800000, 2600000, 2000000, 2100000, 2300000, 1900000,
                     1200000, 1400000, 1300000, 1800000, 2100000, 1300000, 1300000,
                     1800000, 1300000, 1300000, 1800000, 1300000, 1300000, 1500000,
                     1100000, 1100000, 800000, 1200000, 900000, 700000, 600000,
                     800000, 700000, 1600000, 1500000, 1000000, 1000000, 800000,
                     700000, 500000, 500000, 800000, 600000, 600000, 500000,
                     800000, 850000, 550000, 600000, 700000, 400000, 700000, 800000],
            'duration': [142, 175, 152, 154, 142, 148, 136, 146, 118, 121, 
                        178, 139, 202, 124, 201, 195, 130, 169, 151, 189,
                        155, 88, 149, 132, 122, 106, 120, 128, 104, 134,
                        165, 180, 149, 120, 156, 116, 119, 129, 99, 126,
                        117, 113, 144, 91, 164, 100, 101, 147, 115, 125]
        }
        
        # Generate more realistic ratings data
        ratings_data = []
        for user_id in range(1, 101):  # 100 users
            # Each user rates 10-30 random movies
            num_ratings = random.randint(10, 30)
            rated_movies = random.sample(range(1, 51), num_ratings)
            
            for movie_id in rated_movies:
                base_rating = movies_data['rating'][movie_id-1]
                # Add some personal variation
                personal_rating = max(0.5, min(5.0, np.random.normal(base_rating/2, 1.0)))
                rating = round(personal_rating * 2) / 2  # Round to nearest 0.5
                
                ratings_data.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': rating,
                    'timestamp': random.randint(1000000000, 1600000000)
                })
        
        # Users data
        users_data = {
            'userId': list(range(1, 101)),
            'age': [random.randint(18, 65) for _ in range(100)],
            'gender': [random.choice(['M', 'F']) for _ in range(100)],
            'occupation': [random.choice(['student', 'engineer', 'doctor', 'teacher', 
                                        'artist', 'scientist', 'writer', 'other']) 
                          for _ in range(100)]
        }
        
        self.movies_df = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data)
        self.users_df = pd.DataFrame(users_data)
        
        print("🎬 Advanced Movie Recommendation System Initialized!")
        print(f"📊 Movies: {len(self.movies_df)}")
        print(f"👥 Users: {len(self.users_df)}")
        print(f"⭐ Ratings: {len(self.ratings_df)}")
        
        # Precompute similarity matrices
        self._precompute_similarities()
    
    def _precompute_similarities(self):
        """Precompute similarity matrices for faster recommendations"""
        print("🔄 Precomputing similarity matrices...")
        
        # Content-based similarity
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        content_features = self.movies_df['genres'] + ' ' + self.movies_df['director']
        self.tfidf_matrix = tfidf.fit_transform(content_features)
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Collaborative filtering setup
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        # KNN model for item-based CF
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_model.fit(self.user_item_matrix.T)  # Item-based
        
        # SVD for matrix factorization
        self.svd_model = TruncatedSVD(n_components=20, random_state=42)
        self.svd_features = self.svd_model.fit_transform(self.user_item_matrix)
        
        print("✅ Similarity matrices computed!")
    
    def content_based_recommendations(self, movie_title, top_n=10):
        """Enhanced content-based recommendations"""
        print(f"\n🎯 Content-Based Recommendations for '{movie_title}'")
        print("=" * 60)
        
        try:
            idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        except IndexError:
            print(f"❌ Movie '{movie_title}' not found in database.")
            return None
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = self.movies_df.iloc[movie_indices][['title', 'genres', 'director', 'rating']]
        recommendations['similarity_score'] = [round(i[1], 3) for i in sim_scores]
        
        # Display with better formatting
        for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
            print(f"{i:2d}. {movie['title']:40} | ⭐ {movie['rating']} | 🎭 {movie['genres']:30} | 👨‍💼 {movie['director']:20} | 🔍 {movie['similarity_score']:.3f}")
        
        return recommendations
    
    def collaborative_filtering_recommendations(self, user_id, top_n=10):
        """Enhanced collaborative filtering with multiple approaches"""
        print(f"\n👥 Collaborative Filtering Recommendations for User {user_id}")
        print("=" * 60)
        
        if user_id not in self.user_item_matrix.index:
            print(f"❌ User {user_id} not found in ratings data.")
            return None
        
        # Method 1: SVD-based recommendations
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.svd_features[user_idx]
        
        # Calculate similarity with all movies
        movie_vectors = self.svd_model.components_.T
        predicted_ratings = user_vector @ movie_vectors.T
        
        # Get unrated movies
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_mask = user_ratings == 0
        unrated_movies = self.user_item_matrix.columns[unrated_mask]
        unrated_predictions = predicted_ratings[unrated_mask]
        
        # Get top recommendations
        top_indices = unrated_predictions.argsort()[::-1][:top_n]
        recommended_movie_ids = unrated_movies[top_indices]
        predicted_scores = unrated_predictions[top_indices]
        
        recommendations = []
        for movie_id, score in zip(recommended_movie_ids, predicted_scores):
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'director': movie_info['director'],
                'predicted_rating': round(score, 2)
            })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['title']:40} | ⭐ {rec['predicted_rating']:4} | 🎭 {rec['genres']:30} | 👨‍💼 {rec['director']:20}")
        
        return pd.DataFrame(recommendations)
    
    def knn_recommendations(self, movie_title, top_n=10):
        """K-Nearest Neighbors recommendations"""
        print(f"\n🔍 KNN Recommendations for '{movie_title}'")
        print("=" * 60)
        
        try:
            movie_id = self.movies_df[self.movies_df['title'] == movie_title]['movieId'].values[0]
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        except (IndexError, KeyError):
            print(f"❌ Movie '{movie_title}' not found.")
            return None
        
        # Find similar movies using KNN
        distances, indices = self.knn_model.kneighbors(
            self.user_item_matrix.T.iloc[movie_idx:movie_idx+1], 
            n_neighbors=top_n+1
        )
        
        similar_movies = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if i == 0:  # Skip the movie itself
                continue
            movie_id = self.user_item_matrix.columns[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            similar_movies.append({
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'director': movie_info['director'],
                'rating': movie_info['rating'],
                'distance': round(dist, 3)
            })
        
        # Display recommendations
        for i, movie in enumerate(similar_movies, 1):
            print(f"{i:2d}. {movie['title']:40} | ⭐ {movie['rating']} | 🎭 {movie['genres']:30} | 👨‍💼 {movie['director']:20} | 📏 {movie['distance']:.3f}")
        
        return pd.DataFrame(similar_movies)
    
    def hybrid_recommendations(self, user_id, movie_title=None, top_n=10):
        """Advanced hybrid recommendations"""
        print(f"\n🌟 Hybrid Recommendations for User {user_id}")
        if movie_title:
            print(f"   Based on interest in: '{movie_title}'")
        print("=" * 60)
        
        all_recommendations = []
        
        # Get collaborative filtering recommendations
        cf_recs = self.collaborative_filtering_recommendations(user_id, top_n*2)
        if cf_recs is not None:
            for _, rec in cf_recs.iterrows():
                all_recommendations.append({
                    'title': rec['title'],
                    'score': rec['predicted_rating'] * 0.7,  # Weight for CF
                    'type': 'collaborative'
                })
        
        # Get content-based recommendations if movie is provided
        if movie_title:
            content_recs = self.content_based_recommendations(movie_title, top_n*2)
            if content_recs is not None:
                for _, rec in content_recs.iterrows():
                    all_recommendations.append({
                        'title': rec['title'],
                        'score': rec['similarity_score'] * 0.3,  # Weight for content-based
                        'type': 'content'
                    })
        
        # Remove duplicates and get user's watched movies
        user_watched = self.get_user_watched_movies(user_id)
        unique_recs = []
        seen_titles = set()
        
        for rec in all_recommendations:
            if rec['title'] not in seen_titles and rec['title'] not in user_watched:
                unique_recs.append(rec)
                seen_titles.add(rec['title'])
        
        # Sort by combined score
        unique_recs.sort(key=lambda x: x['score'], reverse=True)
        final_recommendations = unique_recs[:top_n]
        
        # Display final recommendations
        for i, rec in enumerate(final_recommendations, 1):
            movie_info = self.movies_df[self.movies_df['title'] == rec['title']].iloc[0]
            print(f"{i:2d}. {rec['title']:40} | ⭐ {movie_info['rating']} | 🎭 {movie_info['genres']:30} | 🔧 {rec['type']:12} | 💯 {rec['score']:.3f}")
        
        return final_recommendations
    
    def get_user_watched_movies(self, user_id):
        """Get movies watched by a user"""
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        watched_movies = user_ratings.merge(self.movies_df, on='movieId')
        return watched_movies['title'].tolist()
    
    def get_popular_movies(self, top_n=10, genre=None, year_range=None):
        """Get popular movies with filters"""
        print(f"\n🏆 Popular Movies" + (f" in {genre}" if genre else "") + (f" from {year_range}" if year_range else ""))
        print("=" * 60)
        
        # Calculate movie statistics
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_stats.columns = ['rating_count', 'rating_mean']
        movie_stats = movie_stats.reset_index()
        
        # Merge with movie data
        popular_movies = movie_stats.merge(self.movies_df, on='movieId')
        
        # Apply filters
        if genre:
            popular_movies = popular_movies[popular_movies['genres'].str.contains(genre, case=False, na=False)]
        
        if year_range:
            start_year, end_year = year_range
            popular_movies = popular_movies[(popular_movies['year'] >= start_year) & (popular_movies['year'] <= end_year)]
        
        # Calculate popularity score (weighted average of rating count and mean)
        popular_movies['popularity_score'] = (
            popular_movies['rating_count'] * 0.3 + 
            popular_movies['rating_mean'] * 0.7
        )
        
        # Sort and get top N
        top_popular = popular_movies.sort_values('popularity_score', ascending=False).head(top_n)
        
        # Display results
        for i, (_, movie) in enumerate(top_popular.iterrows(), 1):
            print(f"{i:2d}. {movie['title']:40} | ⭐ {movie['rating_mean']:4} | 👥 {movie['rating_count']:4} ratings | 🎭 {movie['genres']:30} | 📅 {movie['year']}")
        
        return top_popular
    
    def get_user_profile(self, user_id):
        """Enhanced user profile with statistics"""
        print(f"\n👤 User {user_id} Profile")
        print("=" * 60)
        
        if user_id not in self.users_df['userId'].values:
            print(f"❌ User {user_id} not found.")
            return None
        
        user_info = self.users_df[self.users_df['userId'] == user_id].iloc[0]
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if user_ratings.empty:
            print(f"📊 No ratings from user {user_id}")
            return None
        
        # Merge with movie data
        user_movies = user_ratings.merge(self.movies_df, on='movieId')
        
        # Calculate statistics
        avg_rating = user_ratings['rating'].mean()
        favorite_genres = self._get_favorite_genres(user_ratings)
        rating_count = len(user_ratings)
        
        # Display user info
        print(f"📊 Statistics:")
        print(f"   • Average Rating: {avg_rating:.2f} ⭐")
        print(f"   • Movies Rated: {rating_count}")
        print(f"   • Favorite Genres: {', '.join(favorite_genres)}")
        print(f"   • Age: {user_info['age']} | Gender: {user_info['gender']} | Occupation: {user_info['occupation']}")
        
        print(f"\n🎬 Top Rated Movies:")
        top_rated = user_movies.nlargest(5, 'rating')[['title', 'rating', 'genres']]
        for _, movie in top_rated.iterrows():
            print(f"   ⭐ {movie['rating']} - {movie['title']} ({movie['genres']})")
        
        return user_movies
    
    def _get_favorite_genres(self, user_ratings):
        """Extract favorite genres from user ratings"""
        user_movies = user_ratings.merge(self.movies_df, on='movieId')
        all_genres = []
        for genres in user_movies['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = Counter(all_genres)
        return [genre for genre, _ in genre_counts.most_common(3)]
    
    def search_movies(self, query, search_type='title'):
        """Enhanced movie search"""
        print(f"\n🔍 Search Results for '{query}' in {search_type}")
        print("=" * 60)
        
        if search_type == 'title':
            results = self.movies_df[
                self.movies_df['title'].str.contains(query, case=False, na=False)
            ]
        elif search_type == 'genre':
            results = self.movies_df[
                self.movies_df['genres'].str.contains(query, case=False, na=False)
            ]
        elif search_type == 'director':
            results = self.movies_df[
                self.movies_df['director'].str.contains(query, case=False, na=False)
            ]
        else:
            print("❌ Invalid search type. Use 'title', 'genre', or 'director'.")
            return None
        
        if results.empty:
            print("❌ No movies found matching your search.")
            return None
        
        # Display results
        for i, (_, movie) in enumerate(results.iterrows(), 1):
            print(f"{i:2d}. {movie['title']:40} | ⭐ {movie['rating']} | 🎭 {movie['genres']:30} | 👨‍💼 {movie['director']:20} | 📅 {movie['year']}")
        
        return results
    
    def get_movie_stats(self):
        """Display system statistics"""
        print("\n📈 System Statistics")
        print("=" * 60)
        
        print(f"🎬 Total Movies: {len(self.movies_df)}")
        print(f"👥 Total Users: {len(self.users_df)}")
        print(f"⭐ Total Ratings: {len(self.ratings_df)}")
        print(f"📊 Average Rating: {self.ratings_df['rating'].mean():.2f}")
        print(f"🎭 Unique Genres: {len(set('|'.join(self.movies_df['genres']).split('|')))}")
        print(f"👨‍💼 Unique Directors: {self.movies_df['director'].nunique()}")
        
        # Most popular genres
        all_genres = []
        for genres in self.movies_df['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = Counter(all_genres)
        print(f"\n🏆 Top 5 Genres:")
        for genre, count in genre_counts.most_common(5):
            print(f"   • {genre}: {count} movies")
        
        # Rating distribution
        print(f"\n⭐ Rating Distribution:")
        rating_dist = self.ratings_df['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"   • {rating} stars: {count} ratings")
    
    def visualize_recommendations(self, recommendations, title="Recommendations"):
        """Create visualization for recommendations"""
        if not recommendations:
            print("❌ No recommendations to visualize.")
            return
        
        if isinstance(recommendations, pd.DataFrame):
            # For DataFrame recommendations
            plt.figure(figsize=(12, 8))
            
            if 'similarity_score' in recommendations.columns:
                # Content-based recommendations
                plt.subplot(1, 2, 1)
                plt.barh(recommendations['title'], recommendations['similarity_score'])
                plt.xlabel('Similarity Score')
                plt.title('Content-Based Similarity Scores')
                plt.gca().invert_yaxis()
            
            if 'predicted_rating' in recommendations.columns:
                # Collaborative filtering recommendations
                plt.subplot(1, 2, 2)
                plt.barh(recommendations['title'], recommendations['predicted_rating'])
                plt.xlabel('Predicted Rating')
                plt.title('Predicted Ratings')
                plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.show()

def main():
    """Enhanced main function with beautiful interface"""
    system = AdvancedMovieRecommendationSystem()
    system.generate_comprehensive_data()
    
    print("\n" + "🎬" * 25)
    print("🌟 ADVANCED MOVIE RECOMMENDATION SYSTEM 🌟")
    print("🎬" * 25)
    
    while True:
        print("\n" + "═" * 70)
        print("📋 MAIN MENU")
        print("═" * 70)
        print("1. 🎯 Content-Based Recommendations")
        print("2. 👥 Collaborative Filtering Recommendations") 
        print("3. 🔍 K-Nearest Neighbors Recommendations")
        print("4. 🌟 Hybrid Recommendations")
        print("5. 🏆 Popular Movies")
        print("6. 👤 User Profile")
        print("7. 🔍 Search Movies")
        print("8. 📈 System Statistics")
        print("9. 🎨 Visualize Recommendations")
        print("0. 🚪 Exit")
        print("═" * 70)
        
        choice = input("\n🎯 Enter your choice (0-9): ").strip()
        
        if choice == '1':
            movie_title = input("🎬 Enter movie title: ").strip()
            system.content_based_recommendations(movie_title)
            
        elif choice == '2':
            try:
                user_id = int(input("👤 Enter user ID (1-100): ").strip())
                system.collaborative_filtering_recommendations(user_id)
            except ValueError:
                print("❌ Please enter a valid user ID.")
                
        elif choice == '3':
            movie_title = input("🎬 Enter movie title: ").strip()
            system.knn_recommendations(movie_title)
            
        elif choice == '4':
            try:
                user_id = int(input("👤 Enter user ID (1-100): ").strip())
                movie_title = input("🎬 Enter a movie you like (optional): ").strip()
                system.hybrid_recommendations(user_id, movie_title if movie_title else None)
            except ValueError:
                print("❌ Please enter a valid user ID.")
                
        elif choice == '5':
            print("\n🎭 Filter options:")
            print("   • Leave blank for no filter")
            genre = input("   Enter genre to filter (e.g., Action, Drama): ").strip()
            year_filter = None
            year_input = input("   Enter year range (e.g., 2000-2010): ").strip()
            if year_input and '-' in year_input:
                try:
                    start, end = map(int, year_input.split('-'))
                    year_filter = (start, end)
                except ValueError:
                    print("❌ Invalid year range format.")
            
            system.get_popular_movies(genre=genre if genre else None, year_range=year_filter)
            
        elif choice == '6':
            try:
                user_id = int(input("👤 Enter user ID (1-100): ").strip())
                system.get_user_profile(user_id)
            except ValueError:
                print("❌ Please enter a valid user ID.")
                
        elif choice == '7':
            print("\n🔍 Search by:")
            print("   1. Title")
            print("   2. Genre") 
            print("   3. Director")
            search_choice = input("   Enter choice (1-3): ").strip()
            
            search_types = {'1': 'title', '2': 'genre', '3': 'director'}
            if search_choice in search_types:
                query = input("   Enter search query: ").strip()
                system.search_movies(query, search_types[search_choice])
            else:
                print("❌ Invalid choice.")
                
        elif choice == '8':
            system.get_movie_stats()
            
        elif choice == '9':
            print("🎨 Visualization feature - Run a recommendation first to visualize!")
            
        elif choice == '0':
            print("\n🎉 Thank you for using the Advanced Movie Recommendation System!")
            print("🌟 Happy watching! 🍿")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\n⏎ Press Enter to continue...")

if __name__ == "__main__":
    main()
