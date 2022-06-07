import sys
import fire
import requests
from traceback import format_exc
from pathlib import Path
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


def download_google_drive(id, path):
    """Downloads a content specified by a given id from Google Drive.
    Args:
        id (str): The Google drive identifier.
        path (str): The path to the output file.
    """
    URL = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()
    response = session.get(
        URL,
        params = { 'id' : id },
        stream = True)
    token = get_token(response)
    if token:
        response = session.get(
            URL, 
            params = { 'id' : id, 'confirm' : token },
            stream = True)
    save(response, Path(path))


def get_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save(response, path):
    CHUNK_SIZE = 32768
    progress = tqdm(
        total=int(response.headers.get('content-length', 0)),
        unit="iB",
        unit_scale=True)
    with NamedTemporaryFile() as tmp:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                progress.update(len(chunk))
                tmp.write(chunk)
        tmp.flush()
        with ZipFile(Path(tmp.name), 'r') as zipped:
            path.mkdir(parents=True, exist_ok=True)
            zipped.extractall(path)


if __name__ == '__main__':
    try:
        fire.Fire(download_google_drive)
    except Exception:
        print(format_exc(), file=sys.stderr)
