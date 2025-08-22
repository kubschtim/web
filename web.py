import re
import requests
import concurrent.futures
import json
from typing import Dict, List, Tuple, Optional
import openai
from dataclasses import dataclass
import os
from requests.exceptions import RequestException, Timeout
import time
from tqdm import tqdm
from datetime import datetime
import json
from pathlib import Path
import argparse
import asyncio
import aiohttp
from functools import lru_cache
import hashlib

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Website Content Analyzer with AI capabilities")
    parser.add_argument("--url",
                      help="Base URL to analyze (e.g., https://example.com)")
    parser.add_argument("--words", type=str,
                      help="Search words (comma-separated). If not provided, will prompt for input.")
    parser.add_argument("--max-pages", type=int, default=8,
                      help="Maximum number of pages to analyze with AI (default: 8)")
    parser.add_argument("--timeout", type=int, default=10,
                      help="Request timeout in seconds (default: 10)")
    parser.add_argument("--no-ai", action="store_true",
                      help="Disable AI analysis even if API key is present")
    return parser.parse_args()

# Configuration
args = parse_args()

# Ask for URL every run if not provided via --url
if args.url:
    BASE_URL = args.url.strip()
else:
    BASE_URL = input("Enter base URL (e.g., https://example.com): ").strip()

# Ensure scheme
if not BASE_URL.startswith(("http://", "https://")):
    BASE_URL = "https://" + BASE_URL

MAX_RESULTS = 3
REQUEST_TIMEOUT = args.timeout
MAX_RETRIES = 3
CACHE_DURATION = 3600  # 1 hour cache

# Simple in-memory cache
_cache = {}

def get_cache_key(url: str) -> str:
    """Generate cache key for URL"""
    return hashlib.md5(url.encode()).hexdigest()

def get_cached_content(url: str) -> Optional[str]:
    """Get cached content if available and not expired"""
    cache_key = get_cache_key(url)
    if cache_key in _cache:
        content, timestamp = _cache[cache_key]
        if time.time() - timestamp < CACHE_DURATION:
            return content
    return None

def cache_content(url: str, content: str):
    """Cache content with timestamp"""
    cache_key = get_cache_key(url)
    _cache[cache_key] = (content, time.time())

# OpenAI Configuration
if not args.no_ai:
    openai.api_key = os.getenv('OPENAI_API_KEY')

# Get search words
if args.words:
    SEARCH_WORDS = args.words.split(",")
else:
    search_input = input("Enter search words (separated by space): ")
    SEARCH_WORDS = search_input.split()

@dataclass
class PageAnalysis:
    url: str
    category: str
    topics: List[str]
    sentiment: str
    key_phrases: List[str]
    summary: str
    installer_type: str
    installer_confidence: float

async def fetch_url_async(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch URL content asynchronously"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as response:
            if response.status == 200:
                content = await response.text()
                # Clean the content
                clean_text = re.sub("<[^>]*>", " ", content)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                clean_text = clean_text[:3000] if len(clean_text) > 3000 else clean_text
                return clean_text
            else:
                return f"Error: HTTP {response.status}"
    except Exception as e:
        return f"Error: {str(e)}"

async def fetch_multiple_urls_async(urls: List[str]) -> Dict[str, str]:
    """Fetch multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return dict(zip(urls, results))

def count_words_on_page(url, search_words):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            reg = re.sub("<[^>]*>", " ", response.text)
            words = re.findall(r"\b\w+\b", reg.lower())
            
            # Count each search word
            results = {}
            for word in search_words:
                results[word] = words.count(word.lower())
            return results
        except Timeout:
            if attempt == MAX_RETRIES - 1:
                return {word: 0 for word in search_words}
            time.sleep(1)  # Wait before retry
        except RequestException as e:
            # Only show errors for non-404 responses to reduce noise
            if "404" not in str(e):
                print(f"⚠️  Error accessing {url}: {str(e)}")
            return {word: 0 for word in search_words}

def extract_ceo_from_impressum(base_url):
    """Robustly extract CEO/Managing Director from imprint-like pages."""
    import json
    from urllib.parse import urljoin
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8"
    }

    def fetch(url):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception:
            return None

    # 1) Candidate paths
    candidate_paths = [
        "/impressum", "/imprint", "/legal", "/legal-notice", "/rechtliches",
        "/unternehmen/impressum", "/company/imprint", "/kontakt", "/contact",
        "/impressum.html", "/impressum-und-datenschutz", "/impressum-datenschutz",
    ]
    candidates = [urljoin(base_url, p) for p in candidate_paths]

    # 2) Discover imprint link on homepage by anchor text
    home_html = fetch(base_url)
    if home_html:
        try:
            from bs4 import BeautifulSoup
            s = BeautifulSoup(home_html, "html.parser")
            for a in s.find_all("a"):
                txt = (a.get_text() or "").strip().lower()
                if re.search(r"\b(impressum|imprint|legal|rechtlich|contact|kontakt|legal notice)\b", txt):
                    href = a.get("href")
                    if href:
                        candidates.append(urljoin(base_url, href))
        except Exception:
            pass

    # Deduplicate while preserving order
    seen = set()
    uniq_candidates = []
    for u in candidates:
        if u not in seen:
            uniq_candidates.append(u); seen.add(u)

    # Patterns to find management names
    label_patterns = [
        r"geschäftsführer(?:in|innen)?",
        r"geschäftsführung",
        r"vertret(?:en|ungs)berechtigt(?:e[rn]?)?",
        r"managing director(?:s)?",
        r"ceo",
        r"vorstand",
        r"inhaber",
        r"founder(?:s)?",
    ]
    # Capture up to punctuation/newline after label
    capture_after_label = re.compile(
        r"(?:%s)[:\s]+\s*([^\n\r\|•·;:]{3,100})" % ("|".join(label_patterns)),
        re.IGNORECASE
    )

    def clean_names(chunk):
        # Split by common separators
        parts = re.split(r"\s*(?:,| und | & |/|\|)\s*", chunk)
        out = []
        for p in parts:
            p = re.sub(r"[^\w\säöüßÄÖÜ\-]", "", p).strip()
            if not p: 
                continue
            if any(ch.isdigit() for ch in p): 
                continue
            # Heuristic: 2-4 tokens, mostly letters, allow hyphenated names
            toks = [t for t in p.split() if t]
            if 1 < len(toks) <= 4 and len(p) <= 60:
                out.append(p)
        return out

    def jsonld_names(soup):
        names = []
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
            except Exception:
                continue
            def collect(obj):
                if isinstance(obj, dict):
                    # founder(s)
                    if "founder" in obj:
                        v = obj["founder"]
                        if isinstance(v, dict) and "name" in v:
                            names.append(v["name"])
                        elif isinstance(v, list):
                            for it in v:
                                if isinstance(it, dict) and "name" in it:
                                    names.append(it["name"])
                    if "founders" in obj and isinstance(obj["founders"], list):
                        for it in obj["founders"]:
                            if isinstance(it, dict) and "name" in it:
                                names.append(it["name"])
                    # employee with jobTitle CEO/Managing Director
                    emp = obj.get("employee") or obj.get("employees")
                    emp = emp if isinstance(emp, list) else [emp] if emp else []
                    for e in emp:
                        if isinstance(e, dict):
                            jt = (e.get("jobTitle") or "").lower()
                            if re.search(r"\b(ceo|managing director|geschäftsführer)\b", jt) and "name" in e:
                                names.append(e["name"])
                    for v in obj.values():
                        collect(v)
                elif isinstance(obj, list):
                    for v in obj:
                        collect(v)
            collect(data)
        return [n for n in names if isinstance(n, str) and 2 < len(n) < 60]

    for url in uniq_candidates:
        html = fetch(url)
        if not html:
            continue
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove scripts/styles
            for t in soup(["script", "style", "noscript"]):
                t.decompose()

            # 3) Try JSON-LD first
            jl = jsonld_names(soup)
            if jl:
                return ", ".join(sorted(set(jl)))

            # 4) Text scan with patterns
            text = soup.get_text("\n")
            text_l = text.lower()

            # Direct label-based capture
            m = capture_after_label.search(text_l)
            if m:
                names = clean_names(m.group(1))
                if names:
                    return ", ".join(names)

            # 5) Fallback: search lines near labels
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for i, ln in enumerate(lines):
                low = ln.lower()
                if any(lbl in low for lbl in ["geschäftsführer", "geschäftsführung", "managing director", "ceo", "vorstand", "vertretungs"]):
                    # Look current and next two lines
                    neighborhood = " ".join(lines[i:i+3])
                    names = clean_names(neighborhood)
                    if names:
                        return ", ".join(names)
        except Exception:
            continue

    return "CEO not found"

def is_valid_link(link, base_url):
    # Skip problematic URLs
    if any(skip in link.lower() for skip in [
        '/cdn-cgi/',  # Cloudflare email protection
        'mailto:',    # Email links
        'tel:',       # Phone links
        'javascript:', # JavaScript
        '#',          # Anchors
        'data:',      # Data URLs
        'blob:',      # Blob URLs
        '/wp-admin/', # WordPress admin
        '/wp-content/', # WordPress content
        '/wp-includes/', # WordPress includes
        '/api/',      # API endpoints
        '/admin/',    # Admin areas
        '/login',     # Login pages
        '/logout',    # Logout pages
        '/search',    # Search pages
        '/feed',      # RSS feeds
        '/rss',       # RSS feeds
        '/sitemap',   # Sitemaps
        '/robots.txt', # Robots file
        '/favicon',   # Favicons
        '/apple-touch-icon', # Apple icons
        '/manifest.json', # Web app manifests
        '/sw.js',     # Service workers
        '/_next/',    # Next.js
        '/static/',   # Static files
        '/assets/',   # Assets
        '/images/',   # Images
        '/img/',      # Images
        '/css/',      # CSS files
        '/js/',       # JavaScript files
        '/fonts/',    # Font files
        '/uploads/',  # Uploads
        '/cache/',    # Cache
        '/tmp/',      # Temporary files
    ]):
        return False
    
    # Must be a valid relative or absolute URL
    return (link.startswith("/") or 
            link.startswith(base_url)) and \
            not link.startswith("//") and \
            not link.endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp', '.woff', '.woff2', '.ttf', '.eot', '.pdf', '.zip', '.rar', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'))

def build_url(element, base_url):
    if element.startswith("https:"):
        return element
    elif element.startswith("/"):
        return base_url + element
    return None

def get_page_content_optimized(url: str) -> str:
    """Get page content with caching"""
    # Check cache first
    cached = get_cached_content(url)
    if cached:
        return cached
    
    # Fetch from network
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Clean and cache content
            clean_text = re.sub("<[^>]*>", " ", response.text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            clean_text = clean_text[:3000] if len(clean_text) > 3000 else clean_text
            
            cache_content(url, clean_text)
            return clean_text
            
        except Timeout:
            print(f"⚠️  Timeout accessing {url}, attempt {attempt + 1}/{MAX_RETRIES}")
            if attempt == MAX_RETRIES - 1:
                return f"Error: Timeout after {MAX_RETRIES} attempts"
            time.sleep(1)
        except RequestException as e:
            return f"Error loading content: {str(e)}"

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                print(f"⏳ Rate limit reached, waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        self.calls.append(now)

# Global rate limiter for OpenAI API
api_rate_limiter = RateLimiter(calls_per_minute=50)

def analyze_content_with_ai_optimized(url: str, content: str) -> PageAnalysis:
    """Analyze webpage content using OpenAI API with rate limiting"""
    if args.no_ai or not openai.api_key:
        installer_type, installer_confidence = _classify_installer_from_text(content)
        return PageAnalysis(
            url=url,
            category="Heuristic",
            topics=[],
            sentiment="neutral",
            key_phrases=[],
            summary="Heuristic classification only (no AI).",
            installer_type=installer_type,
            installer_confidence=installer_confidence
        )
    
    try:
        # Apply rate limiting
        api_rate_limiter.wait_if_needed()
        
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
        
        installer_type, installer_confidence = _classify_installer_from_text(content)
        return PageAnalysis(
            url=url,
            category=analysis_data.get("category", "Unknown"),
            topics=analysis_data.get("topics", []),
            sentiment=analysis_data.get("sentiment", "neutral"),
            key_phrases=analysis_data.get("key_phrases", []),
            summary=analysis_data.get("summary", "No summary available"),
            installer_type=installer_type,
            installer_confidence=installer_confidence
        )
        
    except Exception as e:
        installer_type, installer_confidence = _classify_installer_from_text(content)
        return PageAnalysis(
            url=url,
            category="Analysis Error",
            topics=[f"Error: {str(e)}"],
            sentiment="neutral",
            key_phrases=["Analysis failed"],
            summary=f"Could not analyze content: {str(e)}",
            installer_type=installer_type,
            installer_confidence=installer_confidence
        )

def analyze_pages_parallel_optimized(urls: List[str]) -> List[PageAnalysis]:
    """Analyze multiple pages in parallel with optimized batching"""
    def analyze_single_page(url):
        content = get_page_content_optimized(url)
        return analyze_content_with_ai_optimized(url, content)
    
    # Process in smaller batches to avoid overwhelming the API
    batch_size = 3
    all_results = []
    
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        print(f"🔍 Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            batch_results = list(executor.map(analyze_single_page, batch_urls))
            all_results.extend(batch_results)
        
        # Small delay between batches
        if i + batch_size < len(urls):
            time.sleep(1)
    
    return all_results

def export_results(page_analyses: List[PageAnalysis], word_counts: Dict[str, int]):
    """Export analysis results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Export to JSON
    json_results = {
        "timestamp": timestamp,
        "base_url": BASE_URL,
        "word_counts": word_counts,
        "page_analyses": [
            {
                "url": analysis.url,
                "category": analysis.category,
                "topics": analysis.topics,
                "sentiment": analysis.sentiment,
                "key_phrases": analysis.key_phrases,
                "summary": analysis.summary,
                "installer_type": analysis.installer_type,
                "installer_confidence": analysis.installer_confidence
            }
            for analysis in page_analyses
        ]
    }
    
    json_file = output_dir / f"analysis_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    # Export to HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Website Analysis Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2em; }}
            .category {{ background: #f5f5f5; padding: 1em; margin: 1em 0; border-radius: 5px; }}
            .page {{ margin: 1em 0; padding: 1em; border: 1px solid #ddd; border-radius: 5px; }}
            .sentiment.positive {{ color: green; }}
            .sentiment.negative {{ color: red; }}
            .sentiment.neutral {{ color: gray; }}
        </style>
    </head>
    <body>
        <h1>Website Analysis Results</h1>
        <p><strong>Base URL:</strong> {BASE_URL}</p>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Word Counts</h2>
        <ul>
        {"".join(f"<li><strong>{word}:</strong> {count}</li>" for word, count in word_counts.items())}
        </ul>
        
        <h2>Page Analyses</h2>
        {"".join(f'''
        <div class="page">
            <h3><a href="{analysis.url}">{analysis.url}</a></h3>
            <p><strong>Category:</strong> {analysis.category}</p>
            <p><strong>Sentiment:</strong> <span class="sentiment {analysis.sentiment}">{analysis.sentiment}</span></p>
            <p><strong>Topics:</strong> {", ".join(analysis.topics)}</p>
            <p><strong>Key Phrases:</strong> {", ".join(analysis.key_phrases)}</p>
            <p><strong>Summary:</strong> {analysis.summary}</p>
            <p><strong>Installer Type:</strong> {analysis.installer_type} ({analysis.installer_confidence:.2f})</p>
        </div>
        ''' for analysis in page_analyses)}
    </body>
    </html>
    """
    
    html_file = output_dir / f"analysis_{timestamp}.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return json_file, html_file

def _classify_installer_from_text(text: str) -> Tuple[str, float]:
    """Heuristically classify installer type from text.
    Returns (installer_type, confidence). installer_type in {"pv", "heatpump", "both", "unknown"}.
    """
    text_l = text.lower()
    pv_keywords = [
        "photovoltaik", "photovoltaik", "pv-anlage", "pv anlage", "solaranlage",
        "solarpanel", "solarmodul", "solarstrom", "pv", "solar", "wechselrichter",
        "strings", "modulmontage", "einspeisung", "speicher", "stromspeicher"
    ]
    hp_keywords = [
        "wärmepumpe", "waermepumpe", "wärmepumpen", "heat pump", "heizungsbauer",
        "heizlast", "monoblock", "splitgerät", "wärmequelle", "hydraulischer abgleich",
        "wp", "kältemittel", "enthalpie"
    ]

    pv_hits = sum(1 for k in pv_keywords if k in text_l)
    hp_hits = sum(1 for k in hp_keywords if k in text_l)

    if pv_hits == 0 and hp_hits == 0:
        return ("unknown", 0.0)

    if pv_hits > 0 and hp_hits == 0:
        # confidence scaled by hits up to 0.95
        return ("pv", min(0.5 + 0.05 * pv_hits, 0.95))
    if hp_hits > 0 and pv_hits == 0:
        return ("heatpump", min(0.5 + 0.05 * hp_hits, 0.95))

    # both present
    dominance = abs(pv_hits - hp_hits)
    base_conf = 0.6 + 0.04 * dominance
    base_conf = min(base_conf, 0.9)
    if pv_hits > hp_hits:
        return ("pv", base_conf)
    if hp_hits > pv_hits:
        return ("heatpump", base_conf)
    return ("both", 0.65)

# Extract CEO information first
print("=== CEO Information ===")
ceo_info = extract_ceo_from_impressum(BASE_URL)
print(f"CEO/Geschäftsführer: {ceo_info}")

# Load main page with error handling
try:
    response = requests.get(BASE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response_text = response.text
    print(f"✅ Successfully loaded main page: {BASE_URL}")
except requests.exceptions.RequestException as e:
    print(f"❌ Error loading main page {BASE_URL}: {str(e)}")
    print("Please check if the URL is correct and the website is accessible.")
    exit(1)

# Extract & filter Links
subpages = re.findall('href="([^"]+)"', response_text)
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
    # Collect all valid URL's
    urls_to_check = []
    for element in liste:
        full_url = build_url(element, BASE_URL)
        if full_url and is_valid_link(full_url, BASE_URL):
            urls_to_check.append(full_url)
    
    print(f"🔍 Found {len(urls_to_check)} valid URLs to analyze")
    
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
if not args.no_ai and openai.api_key:
    print("\n" + "="*60)
    print("🤖 AI-POWERED CONTENT ANALYSIS")
    print("="*60)
    
    # Get top pages for AI analysis (limit to avoid API costs)
    top_urls = urls_to_check[:args.max_pages]
    print(f"Analyzing {len(top_urls)} pages with AI...")
    
    # Run AI analysis
    page_analyses = analyze_pages_parallel_optimized(top_urls)
else:
    # Heuristic-only analysis when AI disabled or missing API key
    top_urls = urls_to_check[:args.max_pages]
    page_analyses = []
    for u in top_urls:
        content = get_page_content_optimized(u)
        ins_type, ins_conf = _classify_installer_from_text(content)
        page_analyses.append(PageAnalysis(
            url=u,
            category="Heuristic",
            topics=[],
            sentiment="neutral",
            key_phrases=[],
            summary="Heuristic classification only (no AI).",
            installer_type=ins_type,
            installer_confidence=ins_conf
        ))

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
        print(f"   🛠️ Installer: {analysis.installer_type} ({analysis.installer_confidence:.2f})")

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
if not openai.api_key:
    print("💡 Tip: Set OPENAI_API_KEY environment variable for full AI analysis")

if page_analyses:
    print("\n📊 Exporting Results...")
    json_file, html_file = export_results(page_analyses, total_counts)
    print(f"✅ Results exported to:")
    print(f"   📄 JSON: {json_file}")
    print(f"   🌐 HTML: {html_file}")
    # Add after lead scoring
    print(f"\n�� SUMMARY STATISTICS")
    print("=" * 30)
    print(f"   Total pages analyzed: {len(page_analyses)}")
    print(f"   Pages with PV content: {sum(1 for a in page_analyses if a.installer_type == 'pv')}")
    print(f"   Pages with heat pump content: {sum(1 for a in page_analyses if a.installer_type == 'heatpump')}")
    print(f"   Pages with both: {sum(1 for a in page_analyses if a.installer_type == 'both')}")
    print(f"   Average installer confidence: {sum(a.installer_confidence for a in page_analyses) / len(page_analyses):.2f}")