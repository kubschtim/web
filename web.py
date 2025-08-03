import re
import requests
import concurrent.futures

# Configuration
BASE_URL = "https://www.autarc.energy"
SEARCH_WORD = "heizlastberechnung"
MAX_RESULTS = 3

def count_words_on_page(url, search_word):
    try:
        response = requests.get(url)
        response.raise_for_status()
        reg = re.sub("<[^>]*>", " ", response.text)
        words = re.findall(r"\b\w+\b", reg.lower())
        return words.count(search_word.lower())
    except Exception:
        return 0

def is_valid_link(link, base_url):
    return (link.startswith("/") or 
            link.startswith(base_url) and 
            not link.startswith("//") and
            not link.endswith(('.css', '.js', '.png', '.jpg', '.gif', '.ico')))

def build_url(element, base_url):
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

# Count words on sub pages (parallel)
results = []
total_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Collect all URL's
    urls_to_check = []
    for element in liste:
        full_url = build_url(element, BASE_URL)
        if full_url:
            urls_to_check.append(full_url)
    
    # Run parallel
    word_counts = list(executor.map(lambda url: count_words_on_page(url, SEARCH_WORD), urls_to_check))
    
    # Connect results
    for url, count in zip(urls_to_check, word_counts):
        results.append((url, count))
        total_count += count

# Sort top 3 pages with most occurence
top_3 = sorted(results, key=lambda x: x[1], reverse=True)[:MAX_RESULTS]

# print solution
print(f"Top {MAX_RESULTS} Seiten mit den meisten Vorkommen von '{SEARCH_WORD}':")
for i, (url, count) in enumerate(top_3, 1):
    print(f"{i}. {url}: {count} Vorkommen")

print(f"\nGesamtanzahl '{SEARCH_WORD}' auf allen Seiten: {total_count}")
