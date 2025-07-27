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
    if element.startswith("/") and not element.startswith("//"):
        liste.append(element)

response_subpage = requests.get(url + liste[4]).text

print(response_subpage)

#print(subpages)