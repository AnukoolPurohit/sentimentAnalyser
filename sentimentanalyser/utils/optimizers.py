def maybe_update(objs, dest, f):
    for obj in objs:
        for k, v in f(obj).items():
            if k not in dest:
                dest[k] = v


def get_defaults(d):
    return getattr(d, '_defaults', {})


def debias(mom, damp, step):
    return damp * (1 - mom ** step) / (1 - mom)
