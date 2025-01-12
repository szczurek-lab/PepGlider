# encoding
ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
ENCODING = {char: i for i, char in enumerate(ALPHABET, 1)}
ENCODING_REV = {**{0: "<pad>"}, **{i: char for i, char in enumerate(ALPHABET, 1)}}

SEQ_LEN = 25
VOCAB_SIZE = 20
PAD_TOKEN = 0
CLS_TOKEN = VOCAB_SIZE

# peptides
PEXIGANAN = "GIGKFLKKAKKFGKAFVKILKK"
TEMPORIN = "FLPLIGRVFSGIL"

# data loading - amp?
MIN_LENGTH = 0
MAX_LENGTH = 25