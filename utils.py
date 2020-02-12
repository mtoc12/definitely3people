import zipfile

def unzip_data():
    with zipfile.ZipFile('caltech-cs155-2020.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/')

def setup():
    unzip_data()

