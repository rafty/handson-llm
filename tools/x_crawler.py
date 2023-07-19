import logging
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

# logging.basicConfig(
#     format='%(asctime)s %(levelname)s:%(message)s',
#     level=logging.INFO)


class Crawler:

    def __init__(self, urls=[]):
        self.visited_urls = []
        self.urls_to_visit = urls

    def download_url(self, url):
        return requests.get(url).text

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')

            # if path and path.startswith('/'):
            if path and path.endswith('html') and not path.startswith('https://'):
                path = urljoin(url, path)

                # print(f'path: {path}')
                print('.')
                time.sleep(0.2)
                yield path

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def crawl(self, url):
        html = self.download_url(url)
        for url in self.get_linked_urls(url, html):
            self.add_url_to_visit(url)

    def run(self):
        while self.urls_to_visit:
            url = self.urls_to_visit.pop(0)

            # logging.info(f'Crawling: {url}')
            # print(f'Crawling: {url}')

            try:
                self.crawl(url)
            except Exception:

                # logging.exception(f'Failed to crawl: {url}')
                # print(f'Failed to crawl: {url}')
                pass

            finally:
                self.visited_urls.append(url)


if __name__ == '__main__':
    # Crawler(urls=['https://www.imdb.com/']).run()

    url = "https://python.langchain.com/en/latest/"
    crawler = Crawler(urls=[url])
    crawler.run()
    with open('visited_urls.tet', 'w') as f:
        f.write(crawler.visited_urls)
