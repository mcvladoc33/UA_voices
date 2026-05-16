import os
import re
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from unicodedata import normalize
from num2words import num2words

# Повністю блокуємо будь-які спроби виходу в інтернет для Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Локальні шляхи всередині однієї папки проєкту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLETTS_PATH = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
VERBALIZER_PATH = os.path.join(BASE_DIR, "models", "mbart-large-50-verbalization")
PRESET_PATH = os.path.join(BASE_DIR, "voices", "Інна Гелевера.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Запуск системи TTS (використовується: {device.upper()}) ---")

# =====================================================================
# 🔥 ПАТЧІ ДЛЯ WINDOWS ТА 100% ЛОКАЛЬНОГО ЗАПУСКУ (БЕЗ ІНТЕРНЕТУ)
# =====================================================================
import styletts2_inference.models


# 1. Обхід валідації назви репозиторію (дозволяє використовувати прямі локальні шляхи)
def fake_hf_hub_download(repo_id, filename, **kwargs):
    local_file_path = os.path.join(repo_id, filename)
    if os.path.exists(local_file_path):
        return local_file_path
    raise FileNotFoundError(f"Локальний файл моделі не знайдено: {local_file_path}")


styletts2_inference.models.hf_hub_download = fake_hf_hub_download

# 2. Патч кодування для Windows (запобігає UnicodeDecodeError при зчитуванні config.yml)
original_open = open


def utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)


styletts2_inference.models.open = utf8_open
# =====================================================================


# 1. Завантаження та налаштування Вербалізатора тексту (mBART)
print("Завантаження локального вербалізатора тексту mBART...")
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

try:
    tokenizer = MBart50TokenizerFast.from_pretrained(VERBALIZER_PATH, local_files_only=True)
    verbalizer_model = MBartForConditionalGeneration.from_pretrained(VERBALIZER_PATH, local_files_only=True).to(device)
    tokenizer.src_lang = "uk_UA"
    tokenizer.tgt_lang = "uk_UA"
    print("✅ Вербалізатор mBART успішно налаштований на uk_UA.")
except Exception as e:
    print(f"❌ Не вдалося завантажити mBART напряму: {e}. Буде використано базову заміну.")
    verbalizer_model = None


# Функція ШІ-вербалізації чисел відповідно до контексту речення
def ai_verbalize(text_input):
    if verbalizer_model is None:
        return text_input
    inputs = tokenizer(text_input, return_tensors="pt", padding=True).to(device)
    generated_tokens = verbalizer_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["uk_UA"],
        max_length=256
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


# 2. Завантаження моделі голосу StyleTTS2
print("Завантаження локальної моделі голосу StyleTTS2...")
from styletts2_inference.models import StyleTTS2

multi_model = StyleTTS2(hf_path=STYLETTS_PATH, device=device)

# 3. Ініціалізація лінгвістичних модулів української мови
from ukrainian_word_stress import Stressifier, StressSymbol
from ipa_uk import ipa

stressify = Stressifier()

# 4. Автоматичне підключення пресету голосу
if os.path.exists(PRESET_PATH):
    print(f"Успішно завантажено голос: Інна Гелевера")
    style = torch.load(PRESET_PATH, map_location=device)
else:
    print(f"[УВАГА] Не знайдено файл пресету за шляхом: {PRESET_PATH}")
    print("Скрипт буде використовувати базовий дефолтний голос моделі.")
    style = None

print("\n--- Система готова! Задайте речення для озвучення. ---")

while True:
    text = input("\nВведіть текст українською (або 'exit' для виходу): ").strip()

    if text.lower() == 'exit' or not text:
        print("Завершення роботи...")
        break

    try:
        # Крок 1: Текстова передобробка через mBART (відмінювання чисел)
        print("Робота вербалізатора...")
        clean_text = ai_verbalize(text)

        # Крок 2: Додатковий локальний фільтр числівників (алгоритмічний бекап)
        digits = re.findall(r'\d+', clean_text)
        if digits:
            def replace_num(match):
                number = int(match.group(0))
                try:
                    return num2words(number, lang='uk')
                except Exception:
                    return match.group(0)


            clean_text = re.sub(r'\d+', replace_num, clean_text)

        print(f"[Оброблений текст]: {clean_text}")

        # Крок 3: Нормалізація символів юнікоду та розстановка наголосів
        text_normalized = normalize('NFKC', clean_text.replace('+', StressSymbol.CombiningAcuteAccent))
        text_with_stress = stressify(text_normalized)

        # Крок 4: Трансформація тексту у фонетичні токени (IPA)
        ipa_text = ipa(text_with_stress)
        if not ipa_text:
            print("❌ Помилка: Не вдалося конвертувати символи тексту.")
            continue

        # Крок 5: Токенізація та Синтез мовлення
        tokens = multi_model.tokenizer.encode(ipa_text)
        wav = multi_model(tokens, speed=1.0, s_prev=style)
        audio_data = wav.cpu().numpy().flatten()

        # Крок 6: Локальне відтворення звуку через динаміки
        print("🔊 Озвучую...")
        sd.play(audio_data, 24000)
        sd.wait()

    except Exception as e:
        print(f"❌ Помилка під час обробки або генерації: {e}")