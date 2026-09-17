# --- ПОВНЕ ПРИДУШЕННЯ ВСЬОГО ТА ОФЛАЙН РЕЖИМ ---
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.package")
warnings.filterwarnings("ignore")

sys.stderr = open(os.devnull, 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

sys.stderr = sys.__stdout__

import logging
import builtins
import time
import numpy as np
import sounddevice as sd

import styletts2_inference.models
from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier
from ipa_uk import ipa as uk_to_ipa

logging.getLogger().setLevel(logging.ERROR)

# --- ПАТЧІ ДЛЯ ЛОКАЛЬНОГО ЗАВАНТАЖЕННЯ ТА UTF-8 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLETTS_PATH = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
VOICE_PATH = os.path.join(BASE_DIR, "voices", "Інна Гелевера.pt")

def fake_hf_hub_download(repo_id, filename, **kwargs):
    local_file_path = os.path.join(repo_id, filename)
    if os.path.exists(local_file_path):
        return local_file_path
    raise FileNotFoundError(f"Файл моделі не знайдено локально: {local_file_path}")

styletts2_inference.models.hf_hub_download = fake_hf_hub_download

original_open = builtins.open

def utf8_open(*args, **kwargs):
    if len(args) > 1 and 'b' in args[1]:
        return original_open(*args, **kwargs)
    if 'mode' in kwargs and 'b' in kwargs['mode']:
        return original_open(*args, **kwargs)
    kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)

builtins.open = utf8_open
styletts2_inference.models.open = utf8_open


# --- ІНІЦІАЛІЗАЦІЯ ---
device = 'cpu'
stressifier = Stressifier()

model = StyleTTS2(hf_path=STYLETTS_PATH, device=device)
style_vector = torch.load(VOICE_PATH, map_location=device)


# --- ГЕНЕРАЦІЯ ---
def generate_and_play(text):
    text_stressed = stressifier(text)
    text_ipa = uk_to_ipa(text_stressed)

    start_time = time.time()

    tokens = model.tokenizer.encode(text_ipa)

    # Запускаємо без розрахунку градієнтів для максимальної швидкості
    with torch.no_grad():
        wav = model(tokens, speed=1.0, s_prev=style_vector)

    wav = wav.cpu().numpy().flatten()

    wav = np.clip(wav, -0.99, 0.99)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))

    elapsed = time.time() - start_time
    print(f"✅ Готово! (Час: {elapsed:.3f} сек)")

    sd.play(wav.astype(np.float32), 24000)
    sd.wait()


# --- MAIN ---
if __name__ == "__main__":
    print("\n🚀 Система готова. Введіть текст (або 'exit' для виходу):")

    while True:
        try:
            user_input = input("\n>>> ")

            if user_input.lower() == 'exit':
                print("👋 Завершення роботи...")
                break

            if not user_input.strip():
                continue

            generate_and_play(user_input)

        except KeyboardInterrupt:
            print("\n👋 Завершення роботи...")
            break

        except Exception as e:
            print(f"❌ Помилка: {e}")