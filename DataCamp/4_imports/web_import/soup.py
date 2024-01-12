from bs4 import BeautifulSoup
import requests
from urllib.request import urlretrieve
import pandas as pd

url = 'https://www.youtube.com/watch?v=IlQnGkfskrQ'

r = requests.get(url)

html_doc = r.text

soup = BeautifulSoup(html_doc)

pretty_soup = soup.prettify()
#print(pretty_soup)

title = soup.title
print(title)

a_tags = soup.find_all('a')

for link in a_tags:
    print(link.get('href'))







