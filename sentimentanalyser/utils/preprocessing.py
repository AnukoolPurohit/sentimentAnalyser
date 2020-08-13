import html, re
from sentimentanalyser.preprocessing.tokens import TOKENS, spec_tok


# ------------------------------------------------------------------------
# The following functions are functions applied to text Pre-Tokenization |
#-------------------------------------------------------------------------

def sub_br(t):
    "Replace <br /> with \n"

    pattern = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
    return pattern.sub("\n", t)

def add_spaces_to_spec_chars(t):
    "Add spaces around special charachters # and /"
    
    return re.sub(r'([#/])', r' \1 ', t)

def rm_extra_spaces(t):
    "Remove extra spaces"
    
    return re.sub(r' {2,}', ' ', t)

def replace_chrep(t):
    "Replace Multiple character repetitions"
    "cccc -> TK_CHREP 4 c"
    
    def _replace_chrep(m):
        c, cc = m.groups()
        return f" {TOKENS.TK_CHREP} {len(cc)+ 1} {c} "
    
    chrep_pattern = re.compile(r'(\S)(\1{3,})')
    return chrep_pattern.sub(_replace_chrep, t)

def replace_wrep(t):
    "Replace Multiple word repetitions"
    "word word word word -> TK_WREP 4 word"
    
    def _replace_wrep(m):
        w, ww = m.groups()
        return f" {TOKENS.TK_WREP} {len(ww.split())+ 1} {w} "
    
    wrep_pattern = re.compile(r'(\b\w+\W+)(\1{3,})')
    return wrep_pattern.sub(_replace_wrep, t)

def fixup_text(x):
    "Various messy things we've seen in documents"
    
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>',TOKENS.UNK).replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    
    return re1.sub(' ', html.unescape(x))

# ------------------------------------------------------------------------
# The following functions are functions applied to text Post-Tokenization |
#-------------------------------------------------------------------------

def deal_all_caps(tokens):
    res = []
    for token in tokens:
        if token.isupper() and len(token)>1:
            res.append(TOKENS.TK_UP)
            res.append(token.lower())
        else:
            res.append(token)
    return res

def deal_first_cap(tokens):
    res = []
    for token in tokens:
        if token[0].isupper() and len(token)>1 and token[1:].islower():
            res.append(TOKENS.TK_MAJ)
            res.append(token.lower())
        else:
            res.append(token)
    return res

def add_bos_eos(x):
    return [TOKENS.BOS] + x + [TOKENS.EOS]

# ------------------------------------------------------------------------
# The following are some default values                                  |
#-------------------------------------------------------------------------

default_pre_rules = [fixup_text, replace_chrep, replace_wrep,
                     add_spaces_to_spec_chars, rm_extra_spaces, sub_br]

default_post_rules = [deal_first_cap, deal_all_caps, add_bos_eos]

default_spec_tok = spec_tok