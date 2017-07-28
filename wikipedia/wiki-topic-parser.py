import os
import re
import sys
import json
import requests

"""
Downloads an article from the wikipedia api and splits it into sections
"""

if len(sys.argv) > 1:
    articles = []
    title = sys.argv[1]
    base = "https://en.wikipedia.org/"
    api = "w/api.php?format=json&action=query&prop=extracts&explaintext=&titles={}"
    data = requests.get(base + api.format(title)).json()

    if "query" in data:
        q = data["query"]
        if "pages" in q:
            p = q["pages"]
            # need the page id to access the text content
            page_id = list(p.keys())[0]
            if page_id in p:
                if "extract" in p[page_id]:
                    extract = p[page_id]["extract"]
                    sections = re.split(r'=+', extract)
                    for section in sections:
                        if len(section) > 300:
                            articles.append(section)

    dirname = os.path.dirname(__file__)
    filename = "-".join(title.strip().lower().split(' '))
    filepath = os.path.join(dirname, filename)
    with open(filepath, 'w') as f:
        f.write(json.dumps(articles))
else:
    print("Please supply an article title")
