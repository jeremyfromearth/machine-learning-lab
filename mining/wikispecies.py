import requests


# curl 'https://en.wikipedia.org/w/api.php?action=query&list=embeddedin&eititle=Template:Speciesbox&eilimit=500&format=json' -o species.json

req = requests.get('https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=&pageids=1807164')
print(req.json())

req = requests.get('https://en.wikipedia.org/w/api.php?action=query&list=embeddedin&eititle=Template:Speciesbox&eilimit=500&format=json');
print(req.json())


