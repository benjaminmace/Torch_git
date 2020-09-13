from zipfile import ZipFile
import wget
import os

url = 'https://download.pytorch.org/tutorial/data.zip'

if not os.path.exists('./data.zip'):
    wget.download(url, './data.zip')

with ZipFile('data.zip', 'r') as zipObj:
    zipObj.extractall()