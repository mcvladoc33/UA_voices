import os
import re
import json
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from unicodedata import normalize
from num2words import num2words

# =====================================================================
# 🤫 ПРИДУШЕННЯ ВСІХ СИСТЕМНИХ ПОПЕРЕДЖЕНЬ (WARNINGS) ТА ЛОГІВ ШІ
# =====================================================================
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Забороняємо бібліотеці transformers спамити в консоль
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# =====================================================================
# 🔒 100% ОФЛАЙН РЕЖИМ ДЛЯ HUGGING FACE ТА ПАТЧІ ДЛЯ WINDOWS
# =====================================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLETTS_PATH = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
VERBALIZER_PATH = os.path.join(BASE_DIR, "models", "mbart-large-50-verbalization")
PRESET_DIR = os.path.join(BASE_DIR, "voices")
REF_DIR = os.path.join(BASE_DIR, "references")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

os.makedirs(PRESET_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🛠️ Налаштування шляхів до FFmpeg (робимо ДО імпорту pydub!)
FFMPEG_BIN = os.path.join(BASE_DIR, "bin")
os.environ["PATH"] += os.path.pathsep + FFMPEG_BIN

from pydub import AudioSegment

AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- UA_voices [Конфіг-Керування] (Робота на: {device.upper()}) ---")

import styletts2_inference.models


def fake_hf_hub_download(repo_id, filename, **kwargs):
    local_file_path = os.path.join(repo_id, filename)
    if os.path.exists(local_file_path):
        return local_file_path
    raise FileNotFoundError(f"Файл моделі не знайдено локально: {local_file_path}")


styletts2_inference.models.hf_hub_download = fake_hf_hub_download

original_open = open


def utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)


styletts2_inference.models.open = utf8_open
# =====================================================================

# 1. Читання налаштувань з config.json
if not os.path.exists(CONFIG_PATH):
    print("❌ Помилка: Не знайдено файл config.json у корені проєкту! Створіть його.")
    exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

mode = config.get("mode", "1")
speed = config.get("speed", 1.0)
noise_scale = config.get("noise_scale", 0.1)
match_duration = config.get("match_duration", False)
use_verbalizer = config.get("use_verbalizer", True)

# 2. Ініціалізація ШІ Моделей
verbalizer_model = None
tokenizer = None

if use_verbalizer:
    print("⏳ Завантаження лінгвістичного вербалізатора mBART...")
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(VERBALIZER_PATH, local_files_only=True)
        verbalizer_model = MBartForConditionalGeneration.from_pretrained(VERBALIZER_PATH, local_files_only=True).to(
            device)
        tokenizer.src_lang = "uk_UA"
        tokenizer.tgt_lang = "uk_UA"
        print("✅ Вербалізатор mBART успішно завантажено в пам'ять.")
    except Exception as e:
        print(f"⚠️ mBART не завантажено ({e}), працює швидка алгоритмічна заміна.")
        verbalizer_model = None
else:
    print("ℹ️ Вербалізатор mBART вимкнено в config.json (Заощаджено ~2 ГБ ОЗП).")

print("⏳ Завантаження нейромережі синтезу StyleTTS2...")
from styletts2_inference.models import StyleTTS2

multi_model = StyleTTS2(hf_path=STYLETTS_PATH, device=device)

from ukrainian_word_stress import Stressifier, StressSymbol
from ipa_uk import ipa

stressify = Stressifier()


def split_to_parts(text_data):
    split_symbols = '.?!:'
    parts = ['']
    index = 0
    for s in text_data:
        parts[index] += s
        if s in split_symbols and len(parts[index]) > 150:
            index += 1
            parts.append('')
    return [p.strip() for p in parts if p.strip()]


# 3. Підготовка обраного голосу відповідно до конфігурації
style = None
target_duration = None

if mode == "1":
    preset_file = config.get("preset_filename", "Інна Гелевера.pt")
    preset_path = os.path.join(PRESET_DIR, preset_file)
    if not os.path.exists(preset_path):
        print(f"❌ Помилка: Пресет '{preset_file}' не знайдено у папці {PRESET_DIR}")
        exit(1)
    print(f"👤 Режим: Пресет. Автоматично завантажено голос: {preset_file}")
    style = torch.load(preset_path, map_location=device)

elif mode == "2":
    ref_file = config.get("reference_filename", "sample.wav")
    ref_path = os.path.join(REF_DIR, ref_file)
    if not os.path.exists(ref_path):
        print(f"❌ Помилка: Аудіо-референс '{ref_file}' не знайдено у папці {REF_DIR}")
        exit(1)
    print(f"🎭 Режим: Клонування голосу. Аналіз характеристик файлу: {ref_file}")

    style = multi_model.extract_voice_features(ref_path)
    if isinstance(style, list):
        style = style[-1]
    style = style.to(device)

    y, _ = librosa.load(ref_path, sr=24000)
    target_duration = librosa.get_duration(y=y, sr=24000)

else:
    print("❌ Помилка: Невідомий режим (mode) у config.json. Вкажіть '1' або '2'.")
    exit(1)

print("\n🚀 СИСТЕМА ГОТОВА! Просто пишіть ваш текст.")

# 4. Головний робочий цикл програми
while True:
    text = input("\nВведіть текст українською (або 'exit' для виходу): ").strip()
    if text.lower() == 'exit' or not text:
        print("Завершення програми...")
        break

    try:
        # 🧠 Розумне пореченнєве очищення тексту за допомогою mBART
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        processed_sentences = []

        for s in raw_sentences:
            if not s.strip():
                continue

            if use_verbalizer and verbalizer_model and re.search(r'\d+', s):
                inputs = tokenizer(s, return_tensors="pt", padding=True).to(device)
                generated_tokens = verbalizer_model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id["uk_UA"],
                    max_length=len(s) + 40,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                clean_s = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                processed_sentences.append(clean_s.strip())
            else:
                processed_sentences.append(s.strip())

        clean_text = " ".join(processed_sentences)

        # Локальний англо-патч (заміна англійського Advel на укр. Адвел)
        clean_text = re.sub(r'\badvel\b', 'Адвел', clean_text, flags=re.IGNORECASE)

        # Надійний бекап-фільтр для випадкових поодиноких цифр
        if re.search(r'\d+', clean_text):
            clean_text = re.sub(r'\d+', lambda m: num2words(int(m.group(0)), lang='uk'), clean_text)

        print(f"[Текст для ШІ]: {clean_text}")
        parts = split_to_parts(clean_text)
        final_speed = speed

        # Розрахунок Smart Duration для режиму клонування голосу
        if mode == "2" and match_duration and target_duration:
            temp_wavs = []
            for t in parts:
                t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
                ps = ipa(stressify(t_norm))
                if ps:
                    tokens = multi_model.tokenizer.encode(ps)
                    w = multi_model(tokens, speed=1.0, s_prev=style)
                    temp_wavs.append(w)
            if temp_wavs:
                gen_len = sum(len(w) for w in temp_wavs) / 24000
                calc_speed = gen_len / target_duration
                if 0.6 <= calc_speed <= 1.4:
                    final_speed = calc_speed

        # Пофрагментний синтез аудіо
        result_wav = []
        for t in parts:
            t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
            ps = ipa(stressify(t_norm))
            if ps:
                tokens = multi_model.tokenizer.encode(ps)
                current_style = style.clone()
                if noise_scale > 0:
                    current_style += torch.randn_like(current_style) * noise_scale

                wav = multi_model(tokens, speed=final_speed, s_prev=current_style)
                result_wav.append(wav.cpu().numpy().flatten())

        if not result_wav:
            print("❌ Помилка: Не вдалося розпізнати фонет у тексті.")
            continue

        audio_data = np.concatenate(result_wav)

        # Визначення шляхів збереження файлів у теку outputs/
        wav_output_path = os.path.join(OUTPUT_DIR, "output.wav")
        mp3_output_path = os.path.join(OUTPUT_DIR, "output.mp3")

        # Миттєве відтворення аудіо
        print("🔊 Озвучую...")
        sd.play(audio_data, 24000)

        # 💾 Збереження у WAV
        sf.write(wav_output_path, audio_data, 24000)

        # 💾 Конвертація та збереження в MP3 через pydub (із правильним конвертуванням float32 -> int16)
        try:
            clipped_audio = np.clip(audio_data, -1.0, 1.0)
            int16_audio = (clipped_audio * 32767).astype(np.int16)

            audio_segment = AudioSegment(
                int16_audio.tobytes(),
                frame_rate=24000,
                sample_width=2,
                channels=1
            )
            audio_segment.export(mp3_output_path, format="mp3", bitrate="192k")
            print(f"💾 Файли успішно створено в 'outputs/':\n   └─ output.wav\n   └─ output.mp3")
        except Exception as mp3_err:
            print(f"⚠️ Не вдалося зберегти MP3 ({mp3_err}). Переконайся, що папка 'bin' містить ffmpeg.exe.")

        sd.wait()

    except Exception as e:
        print(f"❌ Сталася помилка під час обробки: {e}")