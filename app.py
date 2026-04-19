from flask import Flask, request, jsonify
from utils import *

app = Flask(__name__)

@app.route("/")
def home():
    return "Recommender API is running!"

@app.route("/recommend")
def recommend():
    user_id = int(request.args.get("user_id"))
    
    # candidate movies (simple version)
    movie_list = list(genre_matrix.index)
    
    scores = []
    
    for movie in movie_list[:200]:  # limit for speed
        score = predict_score(user_id, movie)
        scores.append((movie, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    top_movies = [m for m, _ in scores[:5]]
    
    return jsonify({"user_id": user_id, "recommendations": top_movies})

if __name__ == "__main__":
    app.run()