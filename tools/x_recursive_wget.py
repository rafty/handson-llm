from bs4 import BeautifulSoup
import requests

SITE = "https://python.langchain.com/en/latest/"

# lists
urls = []


# function created
def scrape(site):

    print(f'site: {site}')

    r = requests.get(site)

    # converting the text
    s = BeautifulSoup(r.text, "html.parser")

    a = 0
    for i in s.find_all("a"):

        a += 1
        if a == 100:
            print(urls)
            break

        href = i.attrs['href']

        if href.endswith('html') and href.startswith('https://'):

            # print(f'{href}')

            # site = site + href
            site = SITE + href

            # print(site)

            if site not in urls:
                urls.append(site)
                print(site)
                # calling it self
                scrape(site)


if __name__ == "__main__":
    # website to be scrape
    # site = "http://example.webscraping.com//"
    # site = "https://python.langchain.com/en/latest//"
    # _site = "https://python.langchain.com/en/latest/"

    scrape(site=SITE)

