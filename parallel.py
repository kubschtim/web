
#GENERATED AN EXAMPLE WITH CURSOR

import time
import concurrent.futures

def slow_function(url):
    """Simulates a slow request"""
    print(f"Starting: {url}")
    time.sleep(2)  # Simulates 2 seconds wait time
    print(f"Finished: {url}")
    return f"Result from {url}"

# URLs to test
urls = ["url1", "url2", "url3", "url4", "url5"]

print("=== WITHOUT Parallelization (sequential) ===")
start_time = time.time()

for url in urls:
    result = slow_function(url)
    print(f"Result: {result}")

end_time = time.time()
print(f"Time without parallelization: {end_time - start_time:.2f} seconds\n")

print("=== WITH Parallelization (parallel) ===")
start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    # Execute all URLs in parallel
    results = list(executor.map(slow_function, urls))

for result in results:
    print(f"Result: {result}")

end_time = time.time()
print(f"Time with parallelization: {end_time - start_time:.2f} seconds")