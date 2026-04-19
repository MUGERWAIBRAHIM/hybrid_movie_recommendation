import os
import requests
import zipfile
import joblib
import torch

# 🔗 Your GitHub release zip URL
ZIP_URL = "https://github.com/user-attachments/files/26870707/modelfileslens.zip"

ZIP_PATH = "/tmp/models.zip"
EXTRACT_PATH = "/tmp/models"

def download_and_extract():
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH, exist_ok=True)
        
        print("Downloading models zip...")
        r = requests.get(ZIP_URL)
        with open(ZIP_PATH, "wb") as f:
            f.write(r.content)
        
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

download_and_extract()

# 📦 Load models
svdpp_model = joblib.load(f"{EXTRACT_PATH}/svdpp_model.pkl")
xgb_model = joblib.load(f"{EXTRACT_PATH}/xgb_model.pkl")
popularity_dict = joblib.load(f"{EXTRACT_PATH}/popularity.pkl")
genre_matrix = joblib.load(f"{EXTRACT_PATH}/genre_matrix.pkl")
user_profiles = joblib.load(f"{EXTRACT_PATH}/user_profiles.pkl")

# 🔁 NCF
from torch import nn

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

# Load mappings
user_map = joblib.load(f"{EXTRACT_PATH}/user_map.pkl")
movie_map = joblib.load(f"{EXTRACT_PATH}/movie_map.pkl")

model = NCF(len(user_map), len(movie_map))
model.load_state_dict(torch.load(f"{EXTRACT_PATH}/ncf_model.pt", map_location=torch.device('cpu')))
model.eval()