import re
import requests

# Configuration
BASE_URL = "https://www.autarc.energy"
SEARCH_WORD = "heizlastberechnung"
MAX_RESULTS = 3

def count_words_on_page(url, search_word):
    """Zählt Wörter auf einer Seite"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        reg = re.sub("<[^>]*>", " ", response.text)
        words = re.findall(r"\b\w+\b", reg.lower())
        return words.count(search_word.lower())
    except Exception:
        return 0

def is_valid_link(link, base_url):
    """Prüft ob Link gültig ist"""
    return (link.startswith("/") or 
            link.startswith(base_url) and 
            not link.startswith("//") and
            not link.endswith(('.css', '.js', '.png', '.jpg', '.gif', '.ico')))

def build_url(element, base_url):
    """Baut vollständige URL aus Element"""
    if element.startswith("https:"):
        return element
    elif element.startswith("/"):
        return base_url + element
    return None

# Load main page 
response = requests.get(BASE_URL).text

# Extract & filter Links
subpages = re.findall('href="([^"]+)"', response)
liste = []

for element in subpages:
    if is_valid_link(element, BASE_URL):
        liste.append(element)

# Remove duplicates
liste = list(set(liste))
liste.append(BASE_URL)

# Count words on sub pages
results = []
total_count = 0

for element in liste:
    full_url = build_url(element, BASE_URL)
    if full_url is None:
        continue
    
    word_count = count_words_on_page(full_url, SEARCH_WORD)
    results.append((full_url, word_count))
    total_count += word_count

# Sort top 3 pages with most occurence
top_3 = sorted(results, key=lambda x: x[1], reverse=True)[:MAX_RESULTS]

# print solution
print(f"Top {MAX_RESULTS} Seiten mit den meisten Vorkommen von '{SEARCH_WORD}':")
for i, (url, count) in enumerate(top_3, 1):
    print(f"{i}. {url}: {count} Vorkommen")

print(f"\nGesamtanzahl '{SEARCH_WORD}' auf allen Seiten: {total_count}")
