import nltk
# nltk.download('punkt') #uncomment it if you are run the first fime
import pickle
import argparse
from src.utils.load_save import load_file, save_file
from collections import Counter
import string


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(anno_file, threshold):
    """Build a simple vocabulary wrapper."""

    annos = load_file(anno_file)
    print('total QA pairs', len(annos))
    counter = Counter()

    for (qns, ans) in zip(annos['question'], annos['answer']):
        # qns, ans = vqa['question'], vqa['answer']
        # text = qns # qns +' ' +ans
        text = str(qns) + ' '+ str(ans)
        tokens = nltk.tokenize.word_tokenize(text.lower())
        counter.update(tokens)

    counter = sorted(counter.items(), key=lambda item:item[1], reverse=True)
    save_file(dict(counter), 'dataset/VideoQA/word_count.json')
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [item[0] for item in counter if item[1] >= threshold]
    print(len(words))
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

