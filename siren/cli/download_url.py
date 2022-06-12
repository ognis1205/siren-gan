import sys
import fire
import requests
from gzip import GzipFile
from traceback import format_exc
from pathlib import Path
from tqdm import tqdm
from tempfile import NamedTemporaryFile


def download_url(url, name, path):
    """Downloads a content specified by a given id from Google Drive.
    Args:
        url (str): The URL to download.
        path (str): The path to the output file.
    """
    session = requests.Session()
    response = session.get(
        url,
        stream = True,
        allow_redirects = True)
    save(response, name, Path(path))


def save(response, name, path):
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
        path.mkdir(parents=True, exist_ok=True)
        with open(path / name, 'wb') as out, GzipFile(Path(tmp.name)) as zipped:
            out.write(zipped.read())


def main():
    try:
        fire.Fire(download_url)
    except Exception:
        print(format_exc(), file=sys.stderr)
