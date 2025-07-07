
# Update compute_saliency() to accept gaze_scores.json and fuse it
import requests
from bs4 import BeautifulSoup
import spacy
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import os
nlp = spacy.load("en_core_web_sm")

# Step 1: Scrape RSS

def get_ndtv_links():
    rss_url = "https://feeds.feedburner.com/ndtvnews-top-stories"
    res = requests.get(f"https://api.rss2json.com/v1/api.json?rss_url={rss_url}")
    data = res.json()
    return [item['link'] for item in data['items']][:5]

# Step 2: Clean article HTML

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

# Step 3: Preprocess and split

def preprocess_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

# Step 4: Compute saliency + fuse gaze

def compute_attention(sentences, gaze_scores):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    tfidf_scores = tfidf_matrix.sum(axis=1)

    data = []
    for i, sentence in enumerate(sentences):
        doc = nlp(sentence)
        ner_score = len([ent for ent in doc.ents])
        pos_weight = 1.0 if i == 0 else 0.5
        tfidf_score = float(tfidf_scores[i])

        saliency = 0.5 * tfidf_score + 0.3 * ner_score + 0.2 * pos_weight
        saliency_score = round(min(saliency / 10, 1.0), 2)

        gaze_score = gaze_scores[i]["gaze_score"] if i < len(gaze_scores) else 0.0
        attention_score = round(0.6 * gaze_score + 0.4 * saliency_score, 2)

        data.append({
            "text": sentence,
            "saliency_score": saliency_score,
            "gaze_score": gaze_score,
            "attention_score": attention_score
        })
    return data

# Step 5: Save

def save_to_json(data):
    with open("input.json", "w") as f:
        json.dump({"data": data}, f, indent=2)

if __name__ == "__main__":
    print("Full pipeline with gaze integration...")
    links = get_ndtv_links()
    all_sentences = []
    for url in links:
        print(f"{url}")
        raw = clean_article_html(url)
        sents = preprocess_sentences(raw)
        all_sentences.extend(sents)
    if os.path.getsize("gaze_scores.json") == 0:
        raise ValueError("gaze_scores.json is empty. Run the gaze tracker again.")

    if all_sentences:
        if not os.path.exists("gaze_scores.json") or os.path.getsize("gaze_scores.json") == 0:
            print("gaze_scores.json is missing or empty. Aborting.")
            exit()

        with open("gaze_scores.json") as f:
            gaze_scores = json.load(f)
        scored = compute_attention(all_sentences, gaze_scores)
        # ---- T5 summarization logic starts here ----
from transformers import AutoTokenizer, T5ForConditionalGeneration

print("Loading T5 summarizer...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_t5(text):
    input_text = f"summarize: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

output_data = {"summarized": [], "skipped": []}

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
    elif 0.1<= score <= 0.3:
        summary = summarize_t5(text)
        output_data["summarized"].append({
            "text": text,
            "attention_score": score,
            "action": "summarized",
            "summary": summary
        })
    else:
        output_data["skipped"].append({
            "text": text,
            "attention_score": score,
            "action": "skipped",
            "summary": None
        })

# Save final result
with open("output.json", "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved final summarized result to output.json")

       
