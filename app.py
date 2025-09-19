import os, base64, mimetypes, time, pytz
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import torch, spacy, nltk, safetensors.torch
import pandas as pd
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from openai import OpenAI

# ---------------- CONFIG ----------------

# chemins relatifs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model")
CSV_PATH = os.path.join(BASE_DIR, "ordered_disease_treatments.csv")


# ---------------- Initialisation ----------------
nltk.download('stopwords', quiet=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NLP
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


# ---------------- Helpers ----------------

def img_to_data_uri(path: str) -> str:
    if not os.path.exists(path):
        return "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

IMG1 = img_to_data_uri(os.path.join(BASE_DIR, "diagnostic.jpg"))
IMG2 = img_to_data_uri(os.path.join(BASE_DIR, "online.jpg"))
IMG3 = img_to_data_uri(os.path.join(BASE_DIR, "welcome.jpg"))

# --- Mistral ---
os.environ["my_key"] = "ZtUgYH2TPmd29J6rc0oIco8mod9F9ZE7"

client = OpenAI(
    api_key=os.environ["my_key"],
    base_url="https://api.mistral.ai/v1"  # 👈 indispensable
)



def get_local_time():
    tz = pytz.timezone("Europe/Paris")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def generate_sentence_from_keywords(keywords, maladie=""):
    prompt = f"""
    Tu es un assistant médical académique.
    Voici une liste de conseils: {', '.join(keywords)}.
    Ta tâche est d'écrire au moins deux phrases cohérentes qui regroupent ces conseils
    pour un patient atteint de {maladie if maladie else "une maladie donnée"}.
    La réponse doit être naturelle et fluide.
    """

    response = client.chat.completions.create(
        model="mistral-small",   # modèles : mistral-small, mistral-medium, mistral-large
        messages=[{"role": "user", "content": prompt}],
        timeout=60                # évite de tourner indéfiniment
    )

    return response.choices[0].message.content.strip()


disease_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
disease_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
state_dict = safetensors.torch.load_file(os.path.join(MODEL_PATH, "model.safetensors"), device="cpu")
disease_model.load_state_dict(state_dict)
disease_model.eval()
id2label = disease_model.config.id2label

translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_langs = translation_tokenizer.additional_special_tokens

WHISPER_MODEL_NAME = "openai/whisper-small"
print(f"🎙️ Whisper model: {WHISPER_MODEL_NAME}")
asr = pipeline(task="automatic-speech-recognition",
               model=WHISPER_MODEL_NAME,
               device=0 if torch.cuda.is_available() else -1)

df_treatments = pd.read_csv(CSV_PATH)

# ---------------- Traduction ----------------
def translate_text(text, target="eng_Latn"):
    if not text:
        return text
    encoded = translation_tokenizer(text, return_tensors="pt")
    tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids(target),
        max_length=128
    )
    return translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

def translate_to_english(text: str):
    try:
        short = detect(text)
    except LangDetectException:
        short = "en"
    src = "eng_Latn"
    for code in nllb_langs:
        if code.startswith(short):
            src = code
            break
    encoded = translation_tokenizer(text, return_tensors="pt")
    tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=128
    )
    return translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0], src

# ---------- Labels UI ----------
UI_EN = {
    "title": "<h1 style='text-align:center;'>🩺 Intelligent Medical Chatbot</h1>",
    "subtitle": "<p style='text-align:center;'>Academic project — Not a substitute for professional medical advice</p>",
    "your_symptoms": "Your Symptoms",
    "placeholder": "Ex: fever, cough, headache",
    "predict_btn": "Predict the disease",
    "clear_btn": "Clear",
    "audio_label": "🎤 Record / Upload",
    "analyze_btn": "Analyze Audio",
    "back_btn": "Go Back To Main Page",
}

# 🌍 Langues
FLORES_TOP50 = {
    "eng_Latn": "English", "fra_Latn": "Français", "spa_Latn": "Español", "deu_Latn": "Deutsch",
    "ita_Latn": "Italiano", "por_Latn": "Português", "rus_Cyrl": "Русский",
    "zho_Hans": "中文 (简体)", "jpn_Jpan": "日本語", "kor_Hang": "한국어",
    "hin_Deva": "हिन्दी", "arb_Arab": "العربية",
}

def build_lang_labels():
    return {code: f"{name} ({code})" for code, name in FLORES_TOP50.items()}

lang_labels = build_lang_labels()
lang_choices = list(lang_labels.values())

def get_lang_code(display_value):
    for code, label in lang_labels.items():
        if label == display_value:
            return code
    return "eng_Latn"

def build_ui_updates_for_lang(target_code: str):
    if target_code == "eng_Latn":
        texts = UI_EN
    else:
        texts = {k: translate_text(v, target_code) for k, v in UI_EN.items()}
    return (
        gr.update(value=f"<h1>{texts['title']}</h1>"),
        gr.update(value=f"<p>{texts['subtitle']}</p>"),
        gr.update(label=texts["your_symptoms"], placeholder=texts["placeholder"]),
        gr.update(value=texts["predict_btn"]),
        gr.update(value=texts["clear_btn"]),
        gr.update(label=texts["audio_label"]),
        gr.update(value=texts["analyze_btn"]),
        gr.update(value=texts["back_btn"]),
        target_code
    )



# ---------------- Historique ----------------
history = []
def add_to_history(patient_id, mode, text, lang):
    timestamp = get_local_time()
    history.append({
        "Patient": patient_id,
        "Mode": mode,
        "Text": text,
        "Language": lang,
        "DateTime": timestamp
    })
    return pd.DataFrame(history), timestamp


def generate_pdf(patient_id, mode, text, lang, disease, treatments, timestamp):
    filename = os.path.join(BASE_DIR, f"patient_{patient_id}.pdf")
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, height - 50, "Medical Advice Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Date: {timestamp}")  # ✅ utilise la date/heure de l'historique
    c.drawString(50, height - 130, f"Patient ID: {patient_id}")
    c.drawString(50, height - 160, f"Mode: {mode}")
    c.drawString(50, height - 190, f"Language: {lang}")
    c.drawString(50, height - 220, f"Symptoms: {text}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 260, "Detected Disease:")
    c.setFont("Helvetica", 12)
    c.drawString(200, height - 260, disease)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 300, "Recommended Treatments:")
    c.setFont("Helvetica", 12)
    y = height - 330
    for treat in treatments.split(";"):
        c.drawString(70, y, f"- {treat.strip()}")
        y -= 20

    c.save()
    return filename


# ---------------- Web Speech helper ----------------
def to_bcp47(src_lang):
    mapping = {
        "eng_Latn": "en-US", "fra_Latn": "fr-FR", "spa_Latn": "es-ES", "deu_Latn": "de-DE",
        "ita_Latn": "it-IT", "por_Latn": "pt-PT", "rus_Cyrl": "ru-RU",
        "zho_Hans": "zh-CN", "jpn_Jpan": "ja-JP", "kor_Hang": "ko-KR",
        "hin_Deva": "hi-IN", "arb_Arab": "ar-SA"
    }
    return mapping.get(src_lang, "en-US")

def render_speak_button(text_to_read, src_lang):
    lang_tag = to_bcp47(src_lang)
    safe_text = (text_to_read or "No advice generated").replace("'", "\\'")
    return f"""
    <div style="margin-top:10px;">
      <button style="padding:8px 14px;border-radius:10px;background:#6a1b9a;color:#fff;border:none;cursor:pointer;font-weight:600;"
        onclick="
          const u = new SpeechSynthesisUtterance('{safe_text}');
          u.lang = '{lang_tag}';
          const voices = speechSynthesis.getVoices();
          if (voices.length > 0) {{
              u.voice = voices.find(v => v.lang === '{lang_tag}') || voices[0];
          }}
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        ">
        🔊 Speak
      </button>
    </div>
    """



# ---------------- Fonctions prédiction ----------------
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    return " ".join(tokens)

def get_precautions_from_csv(disease_name):
    row = df_treatments[df_treatments["disease"].str.lower() == disease_name.lower()]
    if not row.empty:
        return [w.strip() for w in row.iloc[0]["treatments"].split(",") if w.strip()]
    return ["Consult a doctor"]

def predict_and_advise(patient_text_en):
    clean = preprocess_text(patient_text_en)
    if not clean.strip():
        return "⚠️ Empty text after preprocessing.", "None"
    inputs = disease_tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = disease_model(**inputs).logits
        idx = logits.argmax(dim=-1).item()
    disease = id2label[idx]
    treatments = get_precautions_from_csv(disease)
    return disease, "; ".join(treatments)

# ---------------- Fonctions Gradio ----------------
def on_text_submit(symptoms, ui_lang_code):
    if not symptoms or not symptoms.strip():
        html = "<div class='result-box warning'>⚠️ Please enter your symptoms.</div>"
        yield html, pd.DataFrame(history), None
        return

    text_en, src_lang = translate_to_english(symptoms)
    disease_en, treatments_en = predict_and_advise(text_en)

    treatments_fr = translate_text(treatments_en, "fra_Latn")
    disease_fr = translate_text(disease_en, "fra_Latn")

    advice_sentence_fr = generate_sentence_from_keywords(treatments_fr, disease_fr)
    advice_sentence_en = translate_text(advice_sentence_fr, "eng_Latn")

    if src_lang != "eng_Latn":
        disease_out = translate_text(disease_en, src_lang)
        advice_sentence_out = translate_text(advice_sentence_fr, src_lang)
    else:
        disease_out, advice_sentence_out = disease_en, advice_sentence_en

    # ✅ récupérer les deux valeurs
    df_hist, timestamp = add_to_history(len(history)+1, "Text", symptoms, src_lang)
    pdf_path = generate_pdf(len(history), "Text", symptoms, src_lang, disease_out, advice_sentence_out, timestamp)

    # Bouton audio
    speak_btn = render_speak_button(advice_sentence_out, src_lang)
    
    final_text = f"<div class='result-box'>🩺 <b>Disease predicted:</b> {disease_out}<br>💊 <b>Advice:</b> {advice_sentence_out}</div>"

    out = ""
    for ch in final_text:
        out += ch
        yield out, df_hist, pdf_path, *build_ui_updates_for_lang(ui_lang_code)
        time.sleep(0.01)


def on_audio_submit(audio_path, ui_lang_code):
    if audio_path is None:
        html = "<div class='result-box warning'>⚠️ Please record or upload an audio file.</div>"
        return html, pd.DataFrame(history), None, *build_ui_updates_for_lang(ui_lang_code)

    transcription = asr(audio_path)["text"]
    text_en, src_lang = translate_to_english(transcription)
    disease_en, treatments_en = predict_and_advise(text_en)

    treatments_fr = translate_text(treatments_en, "fra_Latn")
    disease_fr = translate_text(disease_en, "fra_Latn")

    advice_sentence_fr = generate_sentence_from_keywords(treatments_fr, disease_fr)
    advice_sentence_en = translate_text(advice_sentence_fr, "eng_Latn")

    if src_lang != "eng_Latn":
        disease_out = translate_text(disease_en, src_lang)
        advice_sentence_out = translate_text(advice_sentence_fr, src_lang)
    else:
        disease_out, advice_sentence_out = disease_en, advice_sentence_en

    # ✅ récupérer les deux valeurs
    df_hist, timestamp = add_to_history(len(history)+1, "Audio", transcription, src_lang)
    pdf_path = generate_pdf(len(history), "Audio", transcription, src_lang, disease_out, advice_sentence_out, timestamp)

    speak_btn = render_speak_button(advice_sentence_out, src_lang)
    result_html = (
        f"<div class='result-box'>🎤 <b>Detected text:</b> {transcription}<br><br>"
        f"🩺 <b>Disease predicted:</b> {disease_out}<br>"
        f"💊 <b>Treatments:</b> {advice_sentence_out}{speak_btn}</div>"
    )

    return result_html, df_hist, pdf_path, *build_ui_updates_for_lang(ui_lang_code)

# ----- Clear -----
def on_clear():
    return "", "", None

def on_audio_clear():
    return None, "", None

# ---------------- Interface Gradio ----------------
with gr.Blocks(css="""
/* ==== Fond global ==== */
html, body, .gradio-container {
    background: linear-gradient(180deg, #ffffff 0%, #f3e5f5 100%);
}
.page-wrap {
    --block-background-fill: #f3e5f5;
    background: var(--block-background-fill, #f3e5f5);
    border-radius: 26px;
    padding: 18px;
    box-shadow: 0 4px 18px rgba(142,36,170,0.15);
}
.page-container {
    max-width: 900px;
    margin: 0 auto;
    background: linear-gradient(145deg, #f8bbd0, #f3e5f5);
    border: 3px solid #8e24aa;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(142,36,170,0.3);
    padding: 40px 30px;
    font-family: 'Segoe UI', sans-serif;
    text-align: center;
}
h1 {
    text-align: center;
    font-size: 2.4em;
    color: #6a1b9a;
    margin: 0 auto 20px auto;
}
.sub-container {
    background: rgba(255,255,255,0.6);
    border: 2px solid #ce93d8;
    border-radius: 20px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 4px 10px rgba(142,36,170,0.15);
}
.carousel {
    position: relative; width: 560px; height: 320px;
    margin: 0 auto 12px auto; overflow: hidden;
}
.carousel img {
    position: absolute; inset: 0; margin: auto;
    max-width: 100%; max-height: 100%;
    border-radius: 16px; box-shadow: 0 4px 10px rgba(0,0,0,.15);
    opacity: 0; animation: fadeSlide 12s infinite;
}
.carousel img:nth-child(1){animation-delay:0s;}
.carousel img:nth-child(2){animation-delay:4s;}
.carousel img:nth-child(3){animation-delay:8s;}
@keyframes fadeSlide{0%{opacity:0;}5%{opacity:1;}28%{opacity:1;}33%{opacity:0;}100%{opacity:0;}}
.start-btn,.back-btn {
    background:linear-gradient(90deg,#8e24aa,#6a1b9a);
    padding:12px 28px; font-size:16px; font-weight:700; border-radius:15px;
    color:#fff; border:none; cursor:pointer;
    box-shadow:0 4px 12px rgba(0,0,0,.2); margin:10px auto; display:block;
}
.back-btn {background:linear-gradient(90deg,#ab47bc,#8e24aa);}
.result-box {
    background: #f1f8e9; border:1px solid #aed581;
    border-radius: 15px; padding: 15px; margin-top: 15px;
    font-size: 16px; color: #33691e; text-align: center;
}
.result-box.warning {
    background: #fff3cd; border:1px solid #ffeeba; color:#856404;
}
""") as demo:

    # ---- PAGE D'ACCUEIL ----
    with gr.Group(visible=True, elem_classes="page-wrap") as accueil:
        gr.HTML(f"""
        <div class='page-container'>
            <h1>🩺 Welcome On Medical Chatbot</h1>
            <div class="carousel">
                <img src="{IMG1}" alt="Diagnostic">
                <img src="{IMG2}" alt="Consultation en ligne">
                <img src="{IMG3}" alt="Bienvenue">
            </div>
            <div class="sub-container">
                <p>You can enter your symptoms in text or simply speak them into your microphone.</p>
            </div>
        """)
        start_btn = gr.Button("Start", elem_classes="start-btn")
        gr.HTML("</div>")

    # ---- PAGE PRINCIPALE ----
    with gr.Group(visible=False, elem_classes="page-wrap") as app_page:
        main_title_html = gr.HTML(UI_EN["title"])
        main_subtitle_html = gr.HTML(UI_EN["subtitle"])

        lang_choice = gr.Dropdown(choices=lang_choices, value=lang_labels["eng_Latn"], label="🌐 Choose your language")
        ui_lang_state = gr.State("eng_Latn")

        with gr.Tabs():
            # Texte
            with gr.TabItem("⌨️ Symptoms in text"):
                with gr.Column(elem_classes="sub-container"):
                    txt_input = gr.Textbox(lines=3, placeholder=UI_EN["placeholder"], label=UI_EN["your_symptoms"])
                    txt_output = gr.HTML()
                    txt_button = gr.Button(UI_EN["predict_btn"])
                    reset_button = gr.Button(UI_EN["clear_btn"])
                    pdf_file = gr.File(label="📄 Download PDF", type="filepath")

            # Audio
            with gr.TabItem("🎤 Symptoms by voice"):
                with gr.Column(elem_classes="sub-container"):
                    audio_input = gr.Audio(sources=["microphone","upload"], type="filepath", label=UI_EN["audio_label"])
                    audio_output = gr.HTML()
                    audio_button = gr.Button(UI_EN["analyze_btn"])
                    audio_reset_button = gr.Button(UI_EN["clear_btn"])
                    pdf_file2 = gr.File(label="📄 Download PDF", type="filepath")

            # Historique
            with gr.TabItem("📜 History"):
                with gr.Column(elem_classes="sub-container"):
                    history_output = gr.Dataframe(
                      headers=["Patient","Mode","Text","Language","DateTime"],
                      wrap=True
                  )

        back_btn = gr.Button(UI_EN["back_btn"], elem_classes="back-btn")

        # ---- Langue ----
        lang_choice.change(
            fn=lambda v: build_ui_updates_for_lang(get_lang_code(v)),
            inputs=lang_choice,
            outputs=[main_title_html, main_subtitle_html,
                     txt_input, txt_button, reset_button,
                     audio_input, audio_button, back_btn, ui_lang_state]
        )

        # ---- Actions ----
        txt_button.click(
            fn=on_text_submit,
            inputs=[txt_input, ui_lang_state],
            outputs=[txt_output, history_output, pdf_file,
                     main_title_html, main_subtitle_html,
                     txt_input, txt_button, reset_button,
                     audio_input, audio_button, back_btn, ui_lang_state]
        )

        audio_button.click(
            fn=on_audio_submit,
            inputs=[audio_input, ui_lang_state],
            outputs=[audio_output, history_output, pdf_file2,
                     main_title_html, main_subtitle_html,
                     txt_input, txt_button, reset_button,
                     audio_input, audio_button, back_btn, ui_lang_state]
        )

        # ---- Boutons Clear ----
        reset_button.click(
            fn=on_clear,
            inputs=[],
            outputs=[txt_input, txt_output, pdf_file]
        )

        audio_reset_button.click(
            fn=on_audio_clear,
            inputs=[],
            outputs=[audio_input, audio_output, pdf_file2]
        )

        # ---- Navigation (retour accueil) ----
        back_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            inputs=None,
            outputs=[accueil, app_page]
        )

    # ---- Navigation (accueil -> app) ----
    start_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [accueil, app_page])

demo.launch(inbrowser=True)