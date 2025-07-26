import re
import requests

url = "https://reonic.com/"
response = requests.get(url).text

#print Request and Response Header
#print("Request-Header:")
#print(response.request.headers)
#print("-----------------------")
#print("Response-Header:")
#print(response.headers)

reg = re.sub("<[^>]*>", " ", response)
words = re.findall(r"\b\w+\b", reg)
count = words.count("Wärmepumpen")
print(count)