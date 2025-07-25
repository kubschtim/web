import requests

url = "https://www.w3schools.com/python/demopage.htm"
response = requests.get(url)

print("Request-Header:")
print(response.request.headers)
print("-----------------------")
print("Response-Header:")
print(response.headers)
