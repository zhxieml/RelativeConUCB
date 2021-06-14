import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as spp
from tqdm import tqdm

from utils.data import get_reduced_concrete_matrix
from utils.data import normalize_relations
from utils.data import save_oneclass

NUM_ARMS = 2000
NUM_USERS = 500
NUM_TRAIN = 100
DIM = 50

np.random.seed(12345)

def load_sparse_matrix(input_folder):
    userID = {}
    artistID = {}
    tagID = {}
    rows = []
    cols = []
    data = []

    artist_to_tagcount = {}
    tag_to_artists = {}
    df = pd.read_csv(os.path.join(input_folder, "user_taggedartists.dat"), sep="\t", header=0, index_col=None)
    for _, row in tqdm(df.iterrows()):
        if row["artistID"] not in artistID:
            artistID[row["artistID"]] = len(artistID)
        if row["userID"] not in userID:
            userID[row["userID"]] = len(userID)
        if row["tagID"] not in tagID:
            tagID[row["tagID"]] = len(tagID)

        artist = artistID[row["artistID"]]
        user = userID[row["userID"]]
        tag = tagID[row["tagID"]]
        cols.append(artist)
        rows.append(user)
        data.append(1)
        if artist not in artist_to_tagcount:
            artist_to_tagcount[artist] = {}
        if tag not in artist_to_tagcount[artist]:
            artist_to_tagcount[artist][tag] = 1
        else:
            artist_to_tagcount[artist][tag] += 1
        if tag not in tag_to_artists:
            tag_to_artists[tag] = []
        else:
            tag_to_artists[tag].append(artist)

    tag_to_artistcount = {}
    for tag in tag_to_artists:
        tag_to_artistcount[tag] = len(set(tag_to_artists[tag]))

    artist_to_tags = {}
    for artist in artist_to_tagcount:
        related_tags = artist_to_tagcount[artist]
        related_tag_to_artistcount = {tag: tag_to_artistcount[tag] for tag in related_tags}

        # Limit the number of tags.
        related_tag_to_artistcount = dict(sorted(related_tag_to_artistcount.items(), key=lambda x: x[1], reverse=True)[:20])
        related_tags = list(related_tag_to_artistcount)
        artist_to_tags[artist] = related_tags

    print(len(tag_to_artists), len(set(rows)), len(set(cols)))
    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), artist_to_tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Path to the input folder.")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Path to the output folder.")
    args = parser.parse_args()

    # Extract data.
    full_matrix, artist_to_tags = load_sparse_matrix(args.input)
    extracted_matrix, artists = get_reduced_concrete_matrix(full_matrix, NUM_USERS, NUM_ARMS)
    extracted_artist_to_tags = {idx: artist_to_tags[artist] for idx, artist in enumerate(artists)}
    extracted_matrix = (extracted_matrix > 0).astype(float)

    # Normalize data.
    extracted_artist_to_tags = normalize_relations(extracted_artist_to_tags)

    # Split train/test.
    train_users = np.random.choice(NUM_USERS, NUM_TRAIN, replace=False)
    test_users = np.arange(NUM_USERS)
    test_users = np.delete(test_users, train_users)
    train_matrix = extracted_matrix[train_users, :]
    test_matrix = extracted_matrix[test_users, :]

    # Save data.
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    np.save(os.path.join(args.output, "arm_to_suparms"), extracted_artist_to_tags)
    np.save(os.path.join(args.output, "rating_train"), train_matrix)
    np.save(os.path.join(args.output, "rating_test"), test_matrix)
    save_oneclass(os.path.join(args.output, "rating_oneclass_train.txt"), train_matrix)
    save_oneclass(os.path.join(args.output, "rating_oneclass_test.txt"), test_matrix)
