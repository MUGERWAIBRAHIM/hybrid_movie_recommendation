from flask import Flask, request
import joblib
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
precomputed = joblib.load("precomputed_recommendations.pkl")
movies = pd.read_csv("movies.csv")

movie_titles = dict(zip(movies["movie_id"], movies["title"]))
movie_genres = dict(zip(movies["movie_id"], movies["genres"]))


# =========================
# HOME ROUTE (IMPORTANT)
# =========================
@app.route("/")
def home():
    return "USSD app is running 🚀"


# =========================
# FORMAT RESPONSE
# =========================
def format_recommendations(user_id):
    if user_id not in precomputed:
        return "No recommendations found."

    top3 = precomputed[user_id][:3]

    message = "Your Movie Picks:\n\n"

    for movie_id, score in top3:
        title = movie_titles.get(movie_id, "Unknown")
        genres = movie_genres.get(movie_id, "Unknown")

        message += f"{title}\n({genres})\n\n"

    # Trim message to avoid USSD length issues
    return message.strip()[:180]


# =========================
# USSD ROUTE
# =========================
@app.route("/ussd", methods=["POST"])
def ussd():

    text = request.values.get("text", "").strip()

    # FIRST SCREEN
    if text == "":
        return "CON Welcome to Movie AI\n1. Get Recommendations"

    # STEP 2
    elif text == "1":
        return "CON Enter your User ID:"

    # STEP 3
    elif text.startswith("1*"):
        parts = text.split("*")

        if len(parts) < 2 or not parts[1].isdigit():
            return "END Invalid User ID."

        user_id = int(parts[1])
        message = format_recommendations(user_id)

        return f"END {message}"

    return "END Invalid option"


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)