from types import SimpleNamespace


spec_tok = "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()
names = "UNK PAD BOS EOS TK_CHREP TK_WREP TK_UP TK_MAJ".split()

TOKENS = SimpleNamespace(**{n:t for n,t in zip(names, spec_tok)})