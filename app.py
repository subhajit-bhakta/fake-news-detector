from flask import Flask, render_template, request, jsonify, session
import os
import re
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FLASK_SECRET = os.getenv("FLASK_SECRET")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", 8))

# CONSTANTS
WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWS_API_URL = "https://newsapi.org/v2/everything"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# THRESHOLDS
WIKI_SIM_THRESHOLD = 0.70
NEWS_SIM_THRESHOLD = 0.65

# INITIALIZATION
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = FLASK_SECRET
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
genai.configure(api_key=GEMINI_API_KEY)

# CLEAN TEXT
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s\.,'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# NEWS VALIDATION
def is_valid_news(text):
    import re

    text = text.strip().lower()

    # 1. Minimum length
    if len(text) < 25:
        return False, "Too short. Please provide a complete news-like statement."

    # 2. Reject casual / greeting messages
    casual_words = [
        "hi", "hello", "hey", "ok", "bro", "dude", "good morning",
        "good night", "help me", "how are you", "what's up"
    ]
    if text in casual_words:
        return False, "Appears to be casual conversation."

    # 3. Reject emojis
    if re.search(r"[😀-🙏🌀🔥❤️-🧿]", text):
        return False, "Contains emojis — not valid news."

    # 4. Must contain ANY meaningful verb (news-like)
    verbs = [
        "is", "was", "were", "has", "have", "had",
        "announced", "reported", "stated", "said", "claims",
        "passes", "died", "kills", "injured", "arrested",
        "launches", "confirms", "denies"
    ]
    if not any(v in text for v in verbs):
        return False, "Missing action words usually present in news."

    # 5. Must contain at least 2 words that look like real nouns/names
    if len([w for w in text.split() if len(w) > 3]) < 3:
        return False, "Not enough meaningful words."

    return True, "Valid news input."

# WIKIPEDIA
def wiki_search(query):
    try:
        params = {
            "action": "query", "list": "search", "srsearch": query,
            "utf8": 1, "format": "json", "srlimit": 3
        }
        r = requests.get(WIKI_SEARCH_URL, params=params, timeout=8)
        data = r.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title")
            summary_resp = requests.get(WIKI_SUMMARY_URL.format(requests.utils.requote_uri(title)), timeout=5)
            if summary_resp.status_code == 200:
                summary = summary_resp.json().get("extract", "")
                if summary:
                    results.append((title, summary))
        return results
    except Exception:
        return []

def wiki_verify(text):
    results = wiki_search(text)
    if not results:
        return False, None, None, 0.0
    texts = [text] + [s for (_, s) in results if s]
    emb = embedder.encode(texts, convert_to_tensor=True)
    cosines = util.cos_sim(emb[0], emb[1:]).cpu().numpy().flatten()
    best_idx = int(np.argmax(cosines))
    best_score = float(cosines[best_idx])
    if best_score >= WIKI_SIM_THRESHOLD:
        title, summary = results[best_idx]
        return True, title, summary, best_score
    return False, None, None, best_score

# NEWS API
def news_verify(query):
    try:
        params = {
            "q": query,
            "language": "en",
            "pageSize": 10,
            "apiKey": NEWS_API_KEY,
        }
        r = requests.get(NEWS_API_URL, params=params, timeout=8)
        data = r.json()

        if data.get("status") != "ok":
            return False, None, 0.0

        articles = data.get("articles", [])
        if not articles:
            return False, None, 0.0

        texts = [query] + [
            f"{a['title']}. {a.get('description','')}" for a in articles
        ]
        emb = embedder.encode(texts, convert_to_tensor=True)
        scores = util.cos_sim(emb[0], emb[1:]).cpu().numpy().flatten()

        idx = int(np.argmax(scores))
        best_score = float(scores[idx])
        article = articles[idx]

        if best_score >= NEWS_SIM_THRESHOLD:
            return True, article, best_score

        return False, None, best_score

    except Exception:
        return False, None, 0.0

# GEMINI CONTEXTUAL REASONING
def gemini_context_analysis(text):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
You are a professional fact-checking AI.
Given the statement:
\"{text}\"
Determine its factual accuracy based on global context, knowledge, and logic.
Respond strictly in one of the following formats:

REAL - <short reason>
FAKE - <short reason>
UNCERTAIN - <short reason>
"""
        response = model.generate_content(
            prompt,
            request_options={"timeout": GEMINI_TIMEOUT}
        )
        return response.text.strip()
    except Exception as e:
        print("Gemini API error:", e)
        return "UNCERTAIN - Gemini reasoning failed"

# GOOGLE FACT CHECK
def factcheck_verify(text):
    try:
        params = {"query": text, "key": GOOGLE_FACTCHECK_API_KEY}
        r = requests.get(FACTCHECK_URL, params=params, timeout=8)
        data = r.json()

        claims = data.get("claims", [])
        if not claims:
            return False, None, None

        claim = claims[0]
        rating = claim.get("claimReview", [{}])[0].get("textualRating", "")
        publisher = claim.get("claimReview", [{}])[0].get("publisher", {}).get("name", "")

        return True, rating, publisher

    except Exception:
        return False, None, None

# ROUTES
@app.route("/")
def home():
    if "history" not in session:
        session["history"] = []
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = clean_text(data.get("text", "").strip())

    # Validation
    ok, reason = is_valid_news(user_text)
    if not ok:
        return jsonify({
            "prediction": "INVALID INPUT",
            "confidence": 0,
            "api_match": reason
        })

    print(f"\n🧠 Analyzing: {user_text}\n{'-'*60}")

    # Initial Defaults
    prediction, confidence, api_match = "Uncertain", 60.0, "No strong match found"

    # 1. Wikipedia Verification
    wiki_ok, wiki_title, wiki_summary, wiki_sim = wiki_verify(user_text)
    if wiki_ok and wiki_sim > WIKI_SIM_THRESHOLD:
        prediction = "Real"
        confidence = wiki_sim * 100
        api_match = f"Wikipedia - {wiki_title}"

    # 2. NewsAPI Verification
    elif True:
        news_ok, article, news_sim = news_verify(user_text)
        if news_ok and news_sim > NEWS_SIM_THRESHOLD:
            src = article.get("source", {}).get("name", "")
            title = article.get("title", "")
            prediction = "Real"
            confidence = news_sim * 100
            api_match = f"{src}: {title}"

        # 3. Gemini Context Analysis
        else:
            ai_result = gemini_context_analysis(user_text)
            if ai_result.upper().startswith("REAL"):
                prediction, confidence = "Real", 90.0
            elif ai_result.upper().startswith("FAKE"):
                prediction, confidence = "Fake", 90.0
            else:
                prediction, confidence = "Uncertain", 65.0
            api_match = ai_result

      # 4. Google Fact Check
    fc_ok, rating, publisher = factcheck_verify(user_text)
    if fc_ok and rating:  # Only override if rating exists
        if "true" in rating.lower():
            prediction = "Real"
            confidence = 95.0
            api_match = f"Google Fact Check ({publisher}: {rating})"

        elif "false" in rating.lower():
            prediction = "Fake"
            confidence = 95.0
            api_match = f"Google Fact Check ({publisher}: {rating})"

        else:
            prediction = "Uncertain"
            confidence = 70.0
            api_match = f"Google Fact Check ({publisher}: {rating})"

    # Save to History
    history = session.get("history", [])
    history.insert(0, {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "text": user_text,
        "prediction": prediction,
        "confidence": confidence,
        "match": api_match
    })
    session["history"] = history[:30]
    session.modified = True

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "api_match": api_match
    })

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("history", None)
    return ("", 204)

if __name__ == "__main__":
    app.run(host="10.44.68.232", port=5000, debug=True)
