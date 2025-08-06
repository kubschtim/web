import re
import requests
import concurrent.futures
import json
from typing import Dict, List, Tuple
import openai
from dataclasses import dataclass
import os

# Configuration
BASE_URL = "https://www.autarc.energy"
MAX_RESULTS = 3

# OpenAI Configuration (set your API key as environment variable)
openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class PageAnalysis:
    url: str
    category: str
    topics: List[str]
    sentiment: str
    key_phrases: List[str]
    summary: str

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

def get_page_content(url: str) -> str:
    """Extract clean text content from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Remove HTML tags and clean up text
        clean_text = re.sub("<[^>]*>", " ", response.text)
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Multiple spaces to single
        clean_text = clean_text.strip()
        
        # Limit text length for API efficiency
        return clean_text[:3000] if len(clean_text) > 3000 else clean_text
    except Exception as e:
        return f"Error loading content: {str(e)}"

def analyze_content_with_ai(url: str, content: str) -> PageAnalysis:
    """Analyze webpage content using OpenAI API"""
    if not openai.api_key:
        return PageAnalysis(
            url=url,
            category="No API Key",
            topics=["API key not configured"],
            sentiment="neutral",
            key_phrases=["Configure OPENAI_API_KEY environment variable"],
            summary="AI analysis unavailable - API key missing"
        )
    
    try:
        prompt = f"""
Analyze this webpage content and provide a structured analysis:

URL: {url}
Content: {content[:2000]}...

Please provide:
1. Main category (e.g., "Technology", "About Us", "Products", "Services", "Contact", etc.)
2. 3-5 key topics discussed
3. Overall sentiment (positive/neutral/negative)
4. 3-5 key phrases or important terms
5. Brief summary (1-2 sentences)

Format your response as JSON:
{{
    "category": "...",
    "topics": ["...", "...", "..."],
    "sentiment": "...",
    "key_phrases": ["...", "...", "..."],
    "summary": "..."
}}
"""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert content analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        # Parse the JSON response
        analysis_data = json.loads(response.choices[0].message.content)
        
        return PageAnalysis(
            url=url,
            category=analysis_data.get("category", "Unknown"),
            topics=analysis_data.get("topics", []),
            sentiment=analysis_data.get("sentiment", "neutral"),
            key_phrases=analysis_data.get("key_phrases", []),
            summary=analysis_data.get("summary", "No summary available")
        )
        
    except Exception as e:
        return PageAnalysis(
            url=url,
            category="Analysis Error",
            topics=[f"Error: {str(e)}"],
            sentiment="neutral",
            key_phrases=["Analysis failed"],
            summary=f"Could not analyze content: {str(e)}"
        )

def analyze_pages_parallel(urls: List[str]) -> List[PageAnalysis]:
    """Analyze multiple pages in parallel"""
    def analyze_single_page(url):
        content = get_page_content(url)
        return analyze_content_with_ai(url, content)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        analyses = list(executor.map(analyze_single_page, urls))
    
    return analyses

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

# 🤖 AI-POWERED CONTENT ANALYSIS
print("\n" + "="*60)
print("🤖 AI-POWERED CONTENT ANALYSIS")
print("="*60)

# Get top pages for AI analysis (limit to avoid API costs)
top_urls = urls_to_check[:8]  # Analyze top 8 pages
print(f"Analyzing {len(top_urls)} pages with AI...")

# Run AI analysis
page_analyses = analyze_pages_parallel(top_urls)

# Group by category
categories = {}
for analysis in page_analyses:
    if analysis.category not in categories:
        categories[analysis.category] = []
    categories[analysis.category].append(analysis)

# Display results by category
print(f"\n📊 FOUND {len(categories)} CATEGORIES:")
print("-" * 40)

for category, pages in categories.items():
    print(f"\n🏷️  {category.upper()} ({len(pages)} pages)")
    print("─" * (len(category) + 15))
    
    for analysis in pages:
        print(f"\n📄 {analysis.url}")
        print(f"   😊 Sentiment: {analysis.sentiment}")
        print(f"   🎯 Topics: {', '.join(analysis.topics[:3])}")
        print(f"   🔑 Key phrases: {', '.join(analysis.key_phrases[:3])}")
        print(f"   📝 Summary: {analysis.summary}")

# Overall sentiment analysis
sentiments = [a.sentiment for a in page_analyses]
sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}

print(f"\n📈 OVERALL WEBSITE SENTIMENT:")
print("-" * 30)
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(sentiments)) * 100
    print(f"   {sentiment.title()}: {count} pages ({percentage:.1f}%)")

# Most common topics across all pages
all_topics = []
for analysis in page_analyses:
    all_topics.extend(analysis.topics)

from collections import Counter
topic_counts = Counter(all_topics)
most_common_topics = topic_counts.most_common(5)

print(f"\n🔥 TOP TOPICS ACROSS WEBSITE:")
print("-" * 30)
for topic, count in most_common_topics:
    print(f"   {topic}: mentioned {count} times")

print(f"\n✨ Analysis complete! Processed {len(page_analyses)} pages.")
print("💡 Tip: Set OPENAI_API_KEY environment variable for full AI analysis")