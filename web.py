import re
import requests
import webbrowser

url = "https://www.autarc.energy"
response = requests.get(url).text

#PRINT Request and Response Header
#-----------------------------------
#print("Request-Header:")
#print(response.request.headers)
#print("-----------------------")
#print("Response-Header:")
#print(response.headers)

#Count occurence of the word "Wärmepumpen" on a website
#-----------------------------------
#reg = re.sub("<[^>]*>", " ", response)
#words = re.findall(r"\b\w+\b", reg)
#count = words.count("Wärmepumpen")

subpages = re.findall('href="([^"]+)"', response)

liste = []

for element in subpages:
    if (element.startswith("/") or element.startswith("https://www.autarc.energy/")) and not element.startswith("//"):
        liste.append(element)

liste.append(url)

def build_url(element):
    if element.startswith("https:"):
        full_url = element
    elif element.startswith("/"):
        full_url = url + element 
    return full_url

results = []
count = 0
for element in liste:
    full_url = build_url(element)
    if full_url is None:
        continue
    try:
        request_subpage = requests.get(full_url).text
    except Exception:
        continue

    
    
        
    #Count occurence of the word "Wärmepumpen" on a website
    #-----------------------------------    
    reg = re.sub("<[^>]*>", " ", request_subpage)
    words = re.findall(r"\b\w+\b", reg.lower())
    count = count + words.count("heizlastberechnung")
    results.append((full_url, words.count("heizlastberechnung")))  # URL und Anzahl speichern

top_3 = sorted(results, key=lambda x: x[1], reverse=True)[:3]
print(top_3)
#print(subpages)
print(count)
