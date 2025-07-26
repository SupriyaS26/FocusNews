# pipeline.py
import requests, re, json, os
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def get_ndtv_links():
    rss_url = "https://feeds.feedburner.com/ndtvnews-top-stories"
    res = requests.get(f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}")
    data = res.json()
    return [item['link'] for item in data['items']][:3]  


def clean_article_html(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(['script', 'style', 'aside', 'footer', 'nav', 'header']):
            tag.decompose()
        content = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
        return re.sub(r'\s+', ' ', content).strip()
    except Exception as e:
        print(f"Error cleaning {url}: {e}")
        return ""

def preprocess_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 30]

def compute_attention(sentences, gaze_scores):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    tfidf_scores = tfidf_matrix.sum(axis=1)

    data = []
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) < 30:
            continue
        doc = nlp(sentence)
        ner_score = len([ent for ent in doc.ents])
        pos_weight = 1.0 if i == 0 else 0.5
        tfidf_score = float(tfidf_scores[i])

        saliency_score = round(min((0.5 * tfidf_score + 0.3 * ner_score + 0.2 * pos_weight) / 10, 1.0), 2)
        gaze_score = gaze_scores[i]["gaze_score"] if i < len(gaze_scores) else 0.0
        attention_score = round(0.6 * gaze_score + 0.4 * saliency_score, 2)

        data.append({
            "text": sentence,
            "saliency_score": saliency_score,
            "gaze_score": gaze_score,
            "attention_score": attention_score
        })
    return data

def summarize_t5_batch(sentences):
    inputs = [f"summarize: {s}" for s in sentences]
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**input_ids, max_length=100, min_length=30, num_beams=4, early_stopping=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def run_pipeline():
    print("Running summarization pipeline...")

    links = get_ndtv_links()
    
    all_sentences = []
    for url in links:
        raw = clean_article_html(url)
        all_sentences.extend(preprocess_sentences(raw))

    if not os.path.exists("gaze_scores.json") or os.path.getsize("gaze_scores.json") == 0:
        raise ValueError("gaze_scores.json is missing or empty.")

    with open("gaze_scores.json") as f:
        gaze_scores = json.load(f)

    scored = compute_attention(all_sentences, gaze_scores)
    to_summarize = [item["text"] for item in scored if 0.1 <= item["attention_score"] <= 0.3][:10]

    summaries = summarize_t5_batch(to_summarize) if to_summarize else []

    output_data = {"summarized": [], "skipped": []}
    summary_index = 0

    for item in scored:
        score = item["attention_score"]
        text = item["text"]

        if score > 0.3:
            output_data["summarized"].append({
                "text": text,
                "attention_score": score,
                "action": "kept_full",
                "summary": None
            })
        elif 0.1 <= score <= 0.3 and summary_index < len(summaries):
            summary = summaries[summary_index]
            if summary.strip():
                output_data["summarized"].append({
                    "text": text,
                    "attention_score": score,
                    "action": "summarized",
                    "summary": summary
                })
            summary_index += 1
        else:
            output_data["skipped"].append({
                "text": text,
                "attention_score": score,
                "action": "skipped",
                "summary": None
            })

    # Filter out blank summaries
    filtered = [a for a in output_data["summarized"] if a["action"] == "kept_full" or (a["summary"] and a["summary"].strip())]

    with open("output.json", "w") as f:
        json.dump({"summarized": filtered, "skipped": output_data["skipped"]}, f, indent=2)

    print("Saved output.json successfully.")

if __name__ == "__main__":
    run_pipeline()
