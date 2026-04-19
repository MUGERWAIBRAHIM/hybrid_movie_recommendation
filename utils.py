import os
import requests
import zipfile
import joblib
import torch
from torch import nn

# 🔗 Your GitHub release zip URL
ZIP_URL = "https://github.com/user-attachments/files/26870707/modelfileslens.zip"

ZIP_PATH = "/tmp/models.zip"
EXTRACT_PATH = "/tmp/models"

# 🔒 Global placeholders
svdpp_model = None
xgb_model = None
popularity_dict = None
genre_matrix = None
user_profiles = None
user_map = None
movie_map = None
model = None

models_loaded = False


# 📥 Download + extract (SAFE)
def download_and_extract():
    if not os.path.exists(EXTRACT_PATH) or len(os.listdir(EXTRACT_PATH)) == 0:
        os.makedirs(EXTRACT_PATH, exist_ok=True)

        print("Downloading models zip...")
        r = requests.get(ZIP_URL)
        r.raise_for_status()  # 🚨 fail fast if download fails

        with open(ZIP_PATH, "wb") as f:
            f.write(r.content)

        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)


# 🔁 NCF model
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user, movie):
        u = self.user_embedding(user)
        m = self.movie_embedding(movie)
        x = torch.cat([u, m], dim=1)
        return self.fc(x).squeeze()


# 🚀 Lazy loader (THE IMPORTANT FIX)
def load_models():
    global svdpp_model, xgb_model, popularity_dict
    global genre_matrix, user_profiles
    global user_map, movie_map, model, models_loaded

    if models_loaded:
        return

    print("Loading models...")

    download_and_extract()

    # 📦 Load classical models
    svdpp_model = joblib.load(f"{EXTRACT_PATH}/svdpp_model.pkl")
    xgb_model = joblib.load(f"{EXTRACT_PATH}/xgb_model.pkl")
    popularity_dict = joblib.load(f"{EXTRACT_PATH}/popularity.pkl")
    genre_matrix = joblib.load(f"{EXTRACT_PATH}/genre_matrix.pkl")
    user_profiles = joblib.load(f"{EXTRACT_PATH}/user_profiles.pkl")

    # 🔁 Load mappings
    user_map = joblib.load(f"{EXTRACT_PATH}/user_map.pkl")
    movie_map = joblib.load(f"{EXTRACT_PATH}/movie_map.pkl")

    # 🤖 Load NCF
    model = NCF(len(user_map), len(movie_map))
    model.load_state_dict(
        torch.load(f"{EXTRACT_PATH}/ncf_model.pt", map_location=torch.device('cpu'))
    )
    model.eval()

    models_loaded = True
    print("Models loaded successfully ✅")
    
def predict_score(user_id, movie_id):
    # Ensure models are loaded
    if not models_loaded:
        load_models()

    # Handle unknown users/movies safely
    if user_id not in user_map or movie_id not in movie_map:
        return 0

    uid = user_map[user_id]
    mid = movie_map[movie_id]

    # 🎯 1. SVD++
    svd_score = svdpp_model.predict(user_id, movie_id).est

    # 🎯 2. Popularity
    pop_score = popularity_dict.get(movie_id, 0)

    # 🎯 3. Genre similarity (simple dot)
    genre_score = 0
    if user_id in user_profiles and movie_id in genre_matrix.index:
        genre_score = (user_profiles[user_id] * genre_matrix.loc[movie_id]).sum()

    # 🎯 4. NCF
    user_tensor = torch.tensor([uid])
    movie_tensor = torch.tensor([mid])
    ncf_score = model(user_tensor, movie_tensor).item()

    # 🎯 5. Combine (simple average or weighted)
    final_score = (
        0.4 * svd_score +
        0.2 * pop_score +
        0.2 * genre_score +
        0.2 * ncf_score
    )

    return float(final_score)