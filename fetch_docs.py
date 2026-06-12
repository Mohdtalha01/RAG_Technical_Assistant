import urllib.request
import os
import ssl

# Bypass SSL certificate verification on macOS Python installs
ssl._create_default_https_context = ssl._create_unverified_context

docs_dir = "/Users/arjun/Desktop/Assignment/rag_assistant/docs"

os.makedirs(docs_dir, exist_ok=True)

urls = [
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/index.md",
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/first-steps.md",
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-params.md",
    "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/query-params.md"
]

for url in urls:
    filename = url.split("/")[-1]
    save_path = os.path.join(docs_dir, f"{url.split('/')[-2]}_{filename}")
    urllib.request.urlretrieve(url, save_path)
    print(f"Downloaded {save_path}")

print("Documentation fetching complete.")
