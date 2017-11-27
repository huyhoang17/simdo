import csv
import logging

from gensim.models import word2vec


FMT_WORD2VEC = "model/{}features_{}context_{}minwords.vec"


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


class MySentences(object):

    def __init__(self, path_file=None):
        self.path_file = path_file

    def __iter__(self):
        if self.path_file is None:
            raise NotImplementedError("Path file data is empty!")
        with open(self.path_file) as f:
            docreader = csv.DictReader(f)
            for row in docreader:
                yield row["text"].split()


def main():

    sentences = MySentences()
    num_features = 100    # Word vector dimensionality
    min_word_count = 7    # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    model = word2vec.Word2Vec(
        sentences, workers=num_workers,
        size=num_features, min_count=min_word_count,
        window=context, sample=downsampling, seed=1
    )
    fmt_name = FMT_WORD2VEC.format(num_features, context, min_word_count)
    model.save(fmt_name)
    del sentences


if __name__ == '__main__':
    main()
