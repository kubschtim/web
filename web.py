import re
import requests
import concurrent.futures

# Configuration
BASE_URL = "https://www.autarc.energy"
MAX_RESULTS = 3

# Ask user for search words
search_input = input("Enter search words (separated by space): ")
SEARCH_WORDS = search_input.split()

def count_words_on_page(url, search_words):
    try:
        response = requests.get(url)
        response.raise_for_status()
        reg = re.sub("<[^>]*>", " ", response.text)
        words = re.findall(r"\b\w+\b", reg.lower())
        
        # Count each search word
        results = {}
        for word in search_words:
            results[word] = words.count(word.lower())
        return results
    except Exception:
        return {word: 0 for word in search_words}

def extract_ceo_from_impressum(base_url):
    """Extracts CEO information from imprint page"""
    try:
        impressum_url = base_url + "/impressum"
        response = requests.get(impressum_url)
        response.raise_for_status()
        
        # Look for CEO patterns in German
        text = response.text.lower()
        
        # Common patterns for CEO in German
        ceo_patterns = [
            r'geschäftsführer[:\s]+([^<\n]+)',
            r'ceo[:\s]+([^<\n]+)',
            r'geschäftsführung[:\s]+([^<\n]+)',
            r'vorstand[:\s]+([^<\n]+)',
            r'geschäftsführer[:\s]*([^<\n]{3,50})',
            r'ceo[:\s]*([^<\n]{3,50})'
        ]
        
        for pattern in ceo_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ceo_name = match.group(1).strip()
                # Clean up the extracted text
                ceo_name = re.sub(r'[^\w\s]', '', ceo_name).strip()
                if len(ceo_name) > 2:  # Only return if we found something meaningful
                    return ceo_name
        
        return "CEO not found"
        
    except Exception as e:
        return f"Could not access imprint page: {str(e)}"

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

# Extract CEO information first
print("=== CEO Information ===")
ceo_info = extract_ceo_from_impressum(BASE_URL)
print(f"CEO/Geschäftsführer: {ceo_info}")

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
total_counts = {word: 0 for word in SEARCH_WORDS}

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Collect all URL's
    urls_to_check = []
    for element in liste:
        full_url = build_url(element, BASE_URL)
        if full_url:
            urls_to_check.append(full_url)
    
    # Run parallel
    word_counts = list(executor.map(lambda url: count_words_on_page(url, SEARCH_WORDS), urls_to_check))
    
    # Connect results
    for url, counts in zip(urls_to_check, word_counts):
        results.append((url, counts))
        for word, count in counts.items():
            total_counts[word] += count

# Sort top 3 pages with most occurence for each word
for word in SEARCH_WORDS:
    print(f"\nTop {MAX_RESULTS} pages with most occurrences of '{word}':")
    
    # Sort by this specific word
    word_results = [(url, counts[word]) for url, counts in results]
    top_3 = sorted(word_results, key=lambda x: x[1], reverse=True)[:MAX_RESULTS]
    
    for i, (url, count) in enumerate(top_3, 1):
        print(f"{i}. {url}: {count} occurrences")
    
    print(f"Total occurrences of '{word}' on all pages: {total_counts[word]}")