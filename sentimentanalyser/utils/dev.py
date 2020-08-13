def get_info(x, pad_id=1):
    mask = (x == pad_id)
    lenghts = x.size(1) - (x == pad_id).sum(1)
    return lenghts, mask

def print_dims(name, tensor):
    print(f'size of {name} is {tensor.shape}')