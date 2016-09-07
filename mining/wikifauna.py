import re
import urllib
import requests
import pandas as pd
from bs4 import BeautifulSoup
from zipfile import ZipFile

# data vars
limit = 120000
new_articles = []
article_titles_saved = set()
article_titles_searched = set()
article_search_space = set(['Giant_squid'])

# base urls
wiki = 'https://en.wikipedia.org/'
api = 'http://en.wikipedia.org/w/api.php'

# taxomomic ranks we are intersted in
taxonomic_rank = [
        'kingdom', 'phylum', 'class', 
        'order', 'family', 'genus', 'species'
    ]

# open the existing data file and add existing titles to list of previously searched titles
try: 
    df = pd.read_csv('./data/fauna.csv.gz', compression='gzip')
    article_titles_searched.update(df['title'].tolist())
    print('Opened existing data file with {} record(s)'.format(len(df)))
except:
    df = pd.DataFrame(columns=['title', 'page-id', 'text'] + taxonomic_rank)
    print('No existing data file found.')

# fetch wikipedia article and scrape out the infobox with taxonomy info
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
        if all(label in taxonomy for label in ('kingdom', 'class', 'order')):
            if taxonomy['kingdom'] == 'animalia':
                return taxonomy

# mine through articles and scrape links to other articles
# save any article that appears to be an animal species along the way        
def go():
    count = 0
    while len(article_search_space) > 0 and count < limit:
        article_title = article_search_space.pop()
        if article_title not in article_titles_searched:
            print('Looking up:', article_title)
            taxonomy = get_taxonomy_for_article(article_title)
            if taxonomy is not None:
                # text
                url = api + '?format=json&action=query&prop=extracts&explaintext=&titles=' + article_title
                req = requests.get(url)
                if req.status_code is not 200: continue
                data = req.json()
                article_text = ''
                try:
                    if 'query' in data and 'pages' in data['query']:
                        node = data['query']['pages']
                        article_page_id = list(node.keys())[0]

                        # clean up the text a bio
                        article_text = data['query']['pages'][article_page_id]['extract']
                        article_text = article_text.split('== References ==')[0]
                        article_text = article_text.split('== Literature ==')[0]

                        # save the new article
                        if article_text is not '':
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
                            new_corpus_size = len(new_articles) + len(df)
                            if new_corpus_size % 10 == 0:
                                print('Corpus size:', new_corpus_size)
                except:
                    print('Error adding', article_title)
            
                # links
                link_count = 0
                if len(article_search_space) < 20000:
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
                    print('Added {} new articles to search space'.format(link_count))

# run the script
# if the script is interupted by ctrl+c, save all data
if __name__ == '__main__':
    try: 
        go()
    except Exception as e:
        print('Process interupted...')
        print(e)
    finally: 
        # merge the new data with the 
        print('Saving')
        new_data = pd.DataFrame(new_articles)
        df = df.append(new_data)
        df = df.drop_duplicates()
        df.to_csv('./data/fauna.csv.gz', compression='gzip', index=False)
        print('Completed with {} records'.format(len(df)))
