import re

def clean_text(text):
    # remove tags
    text=re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # replace special characters and tokens
    text=re.sub("(\\d|\\W)+"," ",text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ", text)
    text = re.sub(r"\)", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\`", "", text)
    return text.strip().lower()
    
def sort_score(tfidf_score):
    tfidf = zip(tfidf_score.col, tfidf_score.data)
    return sorted(tfidf, key=lambda x: (x[1], x[0]), reverse=True)
 
def map_topk_words(feature_names, sorted_score, topk=3): 
    sorted_score = sorted_score[:topk]
    score_vals = []
    feature_vals = []
    
    for idx, score in sorted_score:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results
