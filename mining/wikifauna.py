import re
import urllib
import requests
import pandas as pd
from bs4 import BeautifulSoup
from zipfile import ZipFile

# data vars
count = 0
limit = 100000
new_articles = []
article_titles_saved = set()
article_titles_searched = set()
article_search_space = set()

# base urls
wiki = 'https://en.wikipedia.org/'
api = 'http://en.wikipedia.org/w/api.php'

# taxomomic ranks we are intersted in
taxonomic_rank = [
        'kingdom', 'phylum', 'class', 
        'order', 'family', 'genus', 'species', 'superorder'
    ]

# open the existing data file and add existing titles to list of previously searched titles
try: 
    df = pd.read_csv('./data/fauna.csv.gz', compression='gzip')
    #article_titles_searched.update(df['title'].tolist())
    print('Opened existing data file with {} record(s)'.format(len(df)))
except:
    df = pd.DataFrame(columns=['title', 'page-id', 'text'] + taxonomic_rank)
    print('No existing data file found.')

# get a list of articles that we know have the 'Speciesbox' and seed the search space
print('Mining with limit of {} new records'.format(limit))
req = requests.get(api + '?action=query&list=embeddedin&eititle=Template:Speciesbox&eilimit=500&format=json');
json = req.json()
if 'query' in json and 'embeddedin' in json['query']:
    for obj in json['query']['embeddedin']:
        if obj['title'] and obj['title']:# not in article_titles_searched:
            article_search_space.add(obj['title'])
print('Initial search space has {} records to mine'.format(len(article_search_space)))

# pull up wikipedia article and scrape out the infobox with taxonomy info
def get_taxonomy_for_article(title):
    r = requests.get(wiki + 'wiki/' + title)
    soup = BeautifulSoup(r.content)
    infobox = soup.find_all("table", class_="infobox")
    if len(infobox) > 0:
        taxonomy = {}
        infobox = infobox[0]
        for row in infobox.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 2:
                label = cells[0].text.lower()
                label = re.sub(r'\W+', '', label)
                classification = cells[1].text.lower()
                classification = re.sub(r'\W+', '', classification)
                taxonomy[label] = classification
        if 'kingdom' in taxonomy: 
            if taxonomy['kingdom'] == 'animalia':
                return taxonomy
            
# mine through articles and scrape links to other articles
# saving anything that appears to be an animal species along the way
while len(article_search_space) > 0 and count < limit:
    article_title = article_search_space.pop()
    taxonomy = get_taxonomy_for_article(article_title)
    print('Crawlilng', article_title)
    if taxonomy is not None:
        # text
        url = api + '?format=json&action=query&prop=extracts&explaintext=&titles=' + article_title
        req = requests.get(url)
        if req.status_code is not 200: continue
        data = req.json()
        article_text = ''
        if 'query' in data and 'pages' in data['query']:
            node = data['query']['pages']
            article_page_id = list(node.keys())[0]
            # clean up the text a bit
            article_text = data['query']['pages'][article_page_id]['extract']
            article_text = article_text.split('== References ==')[0]
            article_text = article_text.split('== Literature ==')[0]

        # save the new article
        if article_title not in article_titles_searched:
            article_titles_searched.add(article_title)
            new_article = {'title' : article_title, 'page-id' : article_page_id, 'text' : article_text}
            for rank in taxonomic_rank:
                if rank in taxonomy:
                    new_article[rank] = taxonomy[rank]
                else:
                    new_article[rank] = ''

            count += 1
            new_articles.append(new_article)
            article_titles_saved.add(article_title)
            print('Adding', article_title, 'to corpus')
            print('Corpus will contain', len(new_articles) + len(df))
        
    # links
    link_count = 0
    if len(article_search_space) < 1000:
        url = api + '?action=query&prop=links&format=json&pllimit=500&titles=' + article_title
        req = requests.get(url)
        if req.status_code is not 200: continue
        links = req.json()
        if 'query' in links and 'pages' in links['query']:
            node = links['query']['pages']
            keys = node.keys()
            for key in keys:
                if 'links' in node[key]:
                    for link in node[key]['links']:
                        if 'title' in link:
                            new_title = link['title']
                            if new_title not in article_titles_searched:
                                if 'Template' not in new_title:
                                    link_count += 1
                                    article_search_space.add(link['title'])
                                    print('Adding article to search space', link['title'])

# merge the new data with the 
new_data = pd.DataFrame(new_articles)
df = df.append(new_data)
df.drop_duplicates()
df.to_csv('./data/fauna.csv.gz', compression='gzip', index=False)
print('Completed with {} records'.format(len(df)))
