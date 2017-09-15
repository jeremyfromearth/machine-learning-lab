import os
import re
import sys
import json
import requests

"""
Downloads an article from the wikipedia and tokenizes sentences
"""

if len(sys.argv) > 1:
    title = sys.argv[1].lower()
    base = "https://en.wikipedia.org/"
    api = "w/api.php?format=json&action=query&prop=extracts&explaintext=&titles={}"

    article = {'extract': '', 'url': base + api.format(title), 'sentences': []}
    data = requests.get(article['url']).json()

    if "query" in data:
        q = data["query"]
        if "pages" in q:
            p = q["pages"]
            # need the page id to access the text content
            page_id = list(p.keys())[0]
            if page_id in p:
                extract = p[page_id].get('extract', None)
                article['extract'] = extract
                if extract is not None:
                    article['extract'] = extract
                    sections = re.split(r'=+', extract)
                    for section in sections:
                        s = section.strip()
                        if len(s):
                            sentence_pattern = re.compile(r'([A-Z].*?[\.!?])', re.M)
                            sentences = sentence_pattern.findall(s)
                            for sentence in sentences:
                                if len(sentence) > 50:
                                    article['sentences'].append(sentence)
    
    dirname = os.path.dirname(__file__)
    filename = "-".join(title.strip().lower().split(' ')) + '.json'
    filepath = os.path.join(dirname, filename)
    with open(filepath, 'w') as f:
        f.write(json.dumps(article, indent=2))
else:
    print("Please supply an article title")
