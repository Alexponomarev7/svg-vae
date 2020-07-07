import csv
import urllib.request
from tqdm import tqdm

with open('glyphazzn_urls.txt') as dataset_list:
    dataset_reader = csv.reader(dataset_list, delimiter=',')
    for row in tqdm(dataset_reader):
        glyph_id, glyph_type, glyph_url = [x.strip() for x in row]

        try:
            urllib.request.urlretrieve(glyph_url, glyph_type + "/" + glyph_url.split('/')[-1])
        except Exception as e:
            pass
