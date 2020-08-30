import re


def camel2snake(name):
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()
