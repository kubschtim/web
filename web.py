from asyncio.windows_events import NULL
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

count = 0
for element in liste:
    if element.startswith("https:"):
        response_subpage = requests.get(element).text
    elif element.startswith("/"):
        response_subpage = requests.get(url + element).text
    else:
        continue
    #Count occurence of the word "Wärmepumpen" on a website
    #-----------------------------------    
    reg = re.sub("<[^>]*>", " ", response_subpage)
    words = re.findall(r"\b\w+\b", reg)
    count = count + words.count("Wärmepumpen")
    
#print(count)
print(subpages)