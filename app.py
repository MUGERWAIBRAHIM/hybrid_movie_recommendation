from flask import Flask, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load precomputed recommendations
precomputed = joblib.load("precomputed_recommendations.pkl")

# Load movies metadata
movies = pd.read_csv("movies.csv")

movie_titles = dict(zip(movies["movie_id"], movies["title"]))
movie_genres = dict(zip(movies["movie_id"], movies["genres"]))


def format_recommendations(user_id):
    if user_id not in precomputed:
        return "No recommendations found for this user."

    top3 = precomputed[user_id][:3]

    message = "🎬 Your Movie Picks:\n\n"

    for movie_id, score in top3:
        title = movie_titles.get(movie_id, "Unknown")
        genres = movie_genres.get(movie_id, "Unknown")

        message += f"{title}\n({genres})\n\n"

    return message.strip()


@app.route("/ussd", methods=["POST"])
def ussd():

    text = request.values.get("text", "")

    # FIRST SCREEN
    if text == "":
        return "CON Welcome to Movie AI\n1. Get Recommendations"

    # STEP 2: Ask for user ID
    elif text == "1":
        return "CON Enter your User ID:"

    # STEP 3: User enters ID → text becomes "1*<id>"
    elif text.startswith("1*"):
        try:
            user_id = int(text.split("*")[1])
        except:
            return "END Invalid User ID."

        message = format_recommendations(user_id)

        return f"END {message}"

    return "END Invalid option"


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)