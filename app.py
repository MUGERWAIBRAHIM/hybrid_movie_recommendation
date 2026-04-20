import os
import threading
from flask import Flask, request, jsonify
from utils import load_models, predict_score, models_loaded, genre_matrix

app = Flask(__name__)

# 🔥 Load models in background (NON-BLOCKING)
def background_load():
    load_models()

threading.Thread(target=background_load).start()


@app.route("/")
def home():
    return "Recommender API is running!"


@app.route("/recommend")
def recommend():
    # 🛑 If models not ready yet
    if not models_loaded:
        return jsonify({
            "message": "Models are still loading, please try again shortly."
        }), 503

    # 🛡️ Validate input
    user_id = request.args.get("user_id")
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    user_id = int(user_id)

    # 🎬 Candidate movies
    movie_list = list(genre_matrix.index)

    scores = []

    # ⚡ Limit for performance
    for movie in movie_list[:200]:
        score = predict_score(user_id, movie)
        scores.append((movie, score))

    # 🔝 Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)

    top_movies = [m for m, _ in scores[:5]]

    return jsonify({
        "user_id": user_id,
        "recommendations": top_movies
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)