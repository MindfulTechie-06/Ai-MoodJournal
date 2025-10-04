# app.py
import streamlit as st
import pandas as pd
import datetime
import os
from pathlib import Path
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import io

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="AI Mood Journal", layout="wide", page_icon="📝")
DATA_FILE = "mood_journal.csv"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"  # multi-class emotion model
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # sentiment

# ----------------------------
# Utility functions
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load NLP pipelines (cached)."""
    try:
        sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    except Exception as e:
        st.warning(f"Could not load sentiment model {SENTIMENT_MODEL}: {e}")
        sentiment = pipeline("sentiment-analysis")

    try:
        emotion = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=True)
    except Exception as e:
        st.warning(f"Could not load emotion model {EMOTION_MODEL}: {e}")
        # fallback to text-classification default if custom model fail
        emotion = pipeline("text-classification", return_all_scores=True)

    return sentiment, emotion

def ensure_datafile():
    """Ensure CSV exists and return DataFrame."""
    if not Path(DATA_FILE).exists():
        df = pd.DataFrame(columns=["timestamp", "text", "sentiment_label", "sentiment_score", "top_emotion", "emotion_scores"])
        df.to_csv(DATA_FILE, index=False)
        return df
    else:
        return pd.read_csv(DATA_FILE)

def save_entry(entry: dict):
    df = ensure_datafile()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def analyze_text(text: str, sentiment_pipe, emotion_pipe):
    # Sentiment
    s = sentiment_pipe(text)[0]  # {'label': 'POSITIVE', 'score': 0.99}
    sentiment_label = s.get("label", "")
    sentiment_score = float(s.get("score", 0.0))

    # Emotion: returns list of dicts with scores
    e_all = emotion_pipe(text)[0]  # list of {'label':..., 'score':...}
    # Standardize into label->score mapping
    emotion_scores = {d["label"].lower(): float(d["score"]) for d in e_all}
    # pick top emotion
    top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]

    return sentiment_label, sentiment_score, top_emotion, emotion_scores

def plot_wordcloud(texts):
    text_corpus = " ".join(texts)
    if not text_corpus.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_corpus)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig

def plot_emotion_bar(df):
    counts = df["top_emotion"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Emotion")
    ax.set_title("Emotion Distribution")
    plt.tight_layout()
    return fig

def plot_mood_trend(df):
    # Convert timestamp to datetime and group by date
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df2["date"] = df2["timestamp"].dt.date
    # We'll compute a simple "positivity score" per entry: map sentiment_label to numeric
    def sentiment_numeric(row):
        lab = str(row.get("sentiment_label", "")).lower()
        score = float(row.get("sentiment_score", 0.0))
        if "positive" in lab or lab == "pos":
            return 1 * score
        if "negative" in lab or lab == "neg":
            return -1 * score
        return 0.0
    df2["sent_score_num"] = df2.apply(sentiment_numeric, axis=1)
    daily = df2.groupby("date")["sent_score_num"].mean().reset_index()
    fig, ax = plt.subplots()
    ax.plot(daily["date"], daily["sent_score_num"], marker="o")
    ax.set_ylabel("Avg Sentiment (signed score)")
    ax.set_xlabel("Date")
    ax.set_title("Mood Trend (avg sentiment) over time")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")

def fig_to_image_bytes(fig, fmt="png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf

# ----------------------------
# Load models
# ----------------------------
with st.spinner("Loading models..."):
    sentiment_pipe, emotion_pipe = load_models()

# ----------------------------
# Layout: Sidebar for info / actions
# ----------------------------
st.sidebar.title("AI Mood Journal")
st.sidebar.markdown("""
A simple Streamlit app that analyzes your journal entry for **sentiment** and **emotion**.
- Write how you feel
- Get insights, trends and suggestions
""")
if st.sidebar.button("Open data file (CSV)"):
    st.sidebar.write(f"CSV: `{DATA_FILE}`")

# ----------------------------
# Main UI
# ----------------------------
st.title("📝 AI Mood Journal")
st.markdown("**Track your emotions. Get insights. Feel better.**")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Write your journal entry")
    user_text = st.text_area("How are you feeling? (Write freely)", height=180, placeholder="e.g., I'm nervous about my project presentation tomorrow, but also excited...")
    submit = st.button("Analyze Mood")

    if submit:
        if not user_text.strip():
            st.warning("Please write something first.")
        else:
            # analyze
            with st.spinner("Analyzing..."):
                sentiment_label, sentiment_score, top_emotion, emotion_scores = analyze_text(user_text, sentiment_pipe, emotion_pipe)

            # Save entry
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "text": user_text,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "top_emotion": top_emotion,
                "emotion_scores": str(emotion_scores)
            }
            save_entry(entry)

            # Display results
            st.markdown("### Analysis Result")
            emotion_emoji_map = {
                "joy": "😊", "happy": "😊", "sadness": "😢", "sad": "😢",
                "anger": "😠", "fear": "😨", "surprise": "😮", "disgust": "🤢",
                # fallback general mappings
            }
            emoji = emotion_emoji_map.get(top_emotion.lower(), "🤔")
            st.metric(label="Detected Emotion", value=f"{top_emotion.title()} {emoji}")
            st.write(f"**Sentiment:** {sentiment_label} (score: {sentiment_score:.2f})")

            # Show emotion scores sorted
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            st.write("**Emotion scores:**")
            emotion_df = pd.DataFrame(sorted_emotions, columns=["emotion", "score"])
            st.dataframe(emotion_df)

            # Suggestions mapping
            suggestions = {
                "joy": "You're feeling joyful — keep it up! Consider writing what made you happy today.",
                "happy": "You're feeling happy — celebrate the small wins! Maybe add a gratitude note.",
                "sadness": "It's okay to feel sad. Try deep breathing (4-4-4) and write 3 things you're grateful for.",
                "sad": "Take a small break—call a friend or listen to calm music.",
                "anger": "Try stepping away for 5 minutes, do some physical activity, or practice box breathing.",
                "fear": "Label the fear: what's the worst realistic outcome? Then plan one small step to reduce it.",
                "surprise": "Surprises can be energizing. Reflect on why this surprised you and whether it's positive.",
                "disgust": "Focus on what calms you—maybe a short walk or change of environment.",
            }
            suggestion_text = suggestions.get(top_emotion.lower(), "Take a moment to breathe. Consider writing one small actionable step you can take.")
            st.info(suggestion_text)

            # Optionally show a calming playlist suggestion (YouTube search terms shown; user can search)
            mood_music_map = {
                "sadness": "calming instrumental music",
                "fear": "relaxing ambient music",
                "anger": "calm piano music",
                "joy": "upbeat feel-good songs",
                "surprise": "light upbeat playlist",
            }
            music_suggestion = mood_music_map.get(top_emotion.lower())
            if music_suggestion:
                st.write(f"**Music suggestion:** Try searching for *{music_suggestion}* on your favorite streaming app.")

with col2:
    st.subheader("Quick tips")
    st.markdown("""
    - Be honest — the model learns from your words.
    - Short entries are fine.
    - Use the dashboard to see trends over time.
    """)
    st.subheader("Export / Manage Data")
    df = ensure_datafile()
    st.write(f"Journal entries: **{len(df)}**")
    csv_bytes = to_csv_download(df)
    st.download_button("Download CSV", data=csv_bytes, file_name="mood_journal.csv", mime="text/csv")

# ----------------------------
# Dashboard
# ----------------------------
st.markdown("---")
st.subheader("📊 Mood Dashboard")

df = ensure_datafile()
if df.empty or len(df) == 0:
    st.info("No entries yet — write your first journal entry and click Analyze Mood.")
else:
    # two-column dashboard
    d1, d2 = st.columns([1,1])
    with d1:
        st.markdown("**Emotion distribution**")
        fig_bar = plot_emotion_bar(df)
        st.pyplot(fig_bar)

        st.markdown("**Mood trend (sentiment avg per day)**")
        fig_trend = plot_mood_trend(df)
        st.pyplot(fig_trend)

    with d2:
        st.markdown("**Word cloud**")
        wc_fig = plot_wordcloud(df["text"].tolist())
        if wc_fig is not None:
            st.pyplot(wc_fig)
            img_bytes = fig_to_image_bytes(wc_fig)
            st.download_button("Download wordcloud (PNG)", data=img_bytes, file_name="wordcloud.png", mime="image/png")
        else:
            st.info("Not enough text yet to build a word cloud.")

    st.markdown("**Recent entries**")
    st.dataframe(df.sort_values(by="timestamp", ascending=False).reset_index(drop=True).head(10))

# ----------------------------
# Footer / About
# ----------------------------
st.markdown("---")
st.markdown("""
**About this demo**  
This is a hackathon-ready MVP: simple local CSV storage, Hugging Face pipelines for sentiment & emotion classification, Streamlit UI, and quick visualizations (bar chart, trend line, word cloud).  
""")
st.caption("Built for a 24-hour hackathon • Extend with authentication, persistent DB, multi-user profiles, or personalized interventions.")
