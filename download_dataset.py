import os
import requests
import zipfile

url = 'https://physionet.org/static/published-projects/ptb-xl/ptbxl_database-1.0.3.zip'
local_zip_path = 'ptbxl_database-1.0.3.zip'
response = requests.get(url)
with open(local_zip_path, 'wb') as f:
    f.write(response.content)

with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall('./data_directory')
