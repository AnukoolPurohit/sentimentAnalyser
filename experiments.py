from sentimentanalyser.utils.data import Path, pad_collate, grandparent_splitter, parent_labeler, read_wiki
from sentimentanalyser.utils.files import pickle_dump, pickle_load
from sentimentanalyser.data.text import TextList, SplitData

from sentimentanalyser.preprocessing.processor import TokenizerProcessor, NuemericalizeProcessor

from functools import partial

path_imdb = Path("/home/anukoolpurohit/Documents/AnukoolPurohit/Datasets/imdb")
path_wiki = Path("/home/anukoolpurohit/Documents/AnukoolPurohit/Datasets/wikitext-103")

lm_wiki = pickle_load('dumps/variable/ll_wiki.pickle')

bs, bptt = 32, 70
wiki_data = lm_wiki.lm_databunchify(bs, bptt)

