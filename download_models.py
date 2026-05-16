import os
from huggingface_hub import snapshot_download

# Створюємо локальну папку для моделей всередині проєкту
os.makedirs("./models", exist_ok=True)

print("1. Завантаження моделі голосу StyleTTS2 у локальну теку...")
snapshot_download(
    repo_id="patriotyk/styletts2_ukrainian_multispeaker",
    local_dir="./models/styletts2_ukrainian_multispeaker",
    local_dir_use_symlinks=False
)

print("\n2. Завантаження вербалізатора mBART у локальну теку...")
snapshot_download(
    repo_id="skypro1111/mbart-large-50-verbalization",
    local_dir="./models/mbart-large-50-verbalization",
    local_dir_use_symlinks=False
)

print("\n[УСПІХ] Усі моделі збережено локально в папку ./models/")