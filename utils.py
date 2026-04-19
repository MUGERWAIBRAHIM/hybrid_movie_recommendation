import os
import requests
import joblib
import torch

MODEL_URLS = {
    "svd": "YOUR_URL",
    "xgb": "YOUR_URL",
    "popularity": "YOUR_URL",
    "genre_matrix": "YOUR_URL",
    "user_profiles": "YOUR_URL",
    "ncf": "YOUR_URL",
    "user_map": "YOUR_URL",
    "movie_map": "YOUR_URL"
}

def download_model(name, url):
    path = f"/tmp/{name}"
    if not os.path.exists(path):
        r = requests.get(url)
        open(path, "wb").write(r.content)
    return path

# Load models
svdpp_model = joblib.load(download_model("svd.pkl", MODEL_URLS["svd"]))
xgb_model = joblib.load(download_model("xgb.pkl", MODEL_URLS["xgb"]))
popularity_dict = joblib.load(download_model("pop.pkl", MODEL_URLS["popularity"]))
genre_matrix = joblib.load(download_model("genre.pkl", MODEL_URLS["genre_matrix"]))
user_profiles = joblib.load(download_model("profiles.pkl", MODEL_URLS["user_profiles"]))
ncf_model = joblib.load(download_model("ncf.pkl", MODEL_URLS["ncf"]))
user_map = joblib.load(download_model("user_map.pkl", MODEL_URLS["user_map"]))
movie_map = joblib.load(download_model("movie_map.pkl", MODEL_URLS["movie_map"]))