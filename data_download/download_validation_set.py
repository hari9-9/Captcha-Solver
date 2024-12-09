from urllib.parse import urlparse, parse_qs
import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

save_folder = "downloaded_files"
os.makedirs(save_folder, exist_ok=True)

text_file = "response_output.txt"
username = ""
def download_file(url):
    try:

        url = f'https://cs7ns1.scss.tcd.ie?shortname={username}&myfilename={url}'
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        filename = query_params['myfilename'][0]
        file_path = os.path.join(save_folder, filename)
        

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


with open(text_file, 'r') as infile:
    urls = infile.readlines()
    #urls = [row[0] for row in reader]


while urls:
    failed_links = []
    

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {executor.submit(download_file, url): url for url in urls}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            if not future.result():
                failed_links.append(url)


    urls = failed_links


print("All files downloaded successfully.")
