import argparse
import logging
import os
import re
import time
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd
from flickrapi import FlickrAPI
from tqdm import tqdm

FLICKR_PUBLIC = '90cb7b5c1ea80af5263116dd84219cbc'
FLICKR_SECRET = '55321131ef96714b'


def pull_images(metadata: str, image_dir: str) -> None:
    """
    This function takes a CSV file containing image metadata.  It should contain a column named image_id
    :param metadata: The CSV file containing image metadata
    :param image_dir: Directory to store all images.
    """

    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

    df = pd.read_csv(metadata)
    df['image_id'] = df['image_id'].astype(str)

    done_lines = os.listdir(image_dir)
    done_lines = [re.sub('.jpg', '', x) for x in done_lines]
    pending_lines = list(set(df['image_id'].tolist()) - set(done_lines))

    for row in tqdm(pending_lines):
        image_id = row.strip()
        try:
            file_location = image_dir + image_id + '.jpg'
            image = flickr.photos.getinfo(photo_id=image_id)
            secret = image['photo']['secret']
            server = image['photo']['server']
            farm_id = image['photo']['farm']
            urlretrieve('https://farm%s.staticflickr.com/%s/%s_%s.jpg' % (farm_id, server, image_id, secret),
                        file_location)
            time.sleep(0.2)
        except (KeyError, URLError):
            logging.error('error while processing %s' % (image_id))
    logging.info('Done downloading images')


def main():
    parser = argparse.ArgumentParser(description="Utility to generate images dataset")
    parser.add_argument('--metadata', '-m', required=True, help="CSV file containing image metadata")
    parser.add_argument('--out', '-o', required=True, help="location for images")
    args = parser.parse_args()

    if not os.path.isdir(args.out):
        logging.info("%s does not exist. Creating it." % args.out)
        os.mkdir(args.out)

    pull_images(args.metadata, args.out)


if __name__ == '__main__':
    main()
