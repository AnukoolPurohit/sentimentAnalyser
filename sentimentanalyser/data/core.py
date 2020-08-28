from sentimentanalyser.utils.data import listify


class ListContainer:

    def __init__(self, items):
        self.items = listify(items)
    
    def __getitem__(self, idx):
        try:
            return self.items[idx]
        except TypeError:
            if isinstance(idx[0], bool):
                assert len(idx) == len(self)
                return [v for v, m in zip(self.items, idx) if m]
            return [self.items[i] for i in idx]
    
    def __len__(self):
        return len(self.items)
    
    def __iter__(self):
        return iter(self.items)
    
    def __delitem__(self, idx):
        del(self.items[idx])
    
    def __setitem__(self, idx, value):
        self.items[idx] = value
    
    @staticmethod
    def display_lists(lists):
        res = ""
        for lst in lists:
            res += f"list ({len(lst)} items) {lst[:5].__repr__()[:-1]}…]\t"
        return res

    def __repr__(self):
        if not self.items:
            ret = f"{self.__class__.__name__} ({len(self)} items)\n{self.items}"
        elif isinstance(self.items[0], list):
            disp_lst = self.display_lists(self.items[:10])
            ret = f"{self.__class__.__name__} ({len(self)} items)\n{disp_lst}"
        else:
            ret = f"{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            return f"{ret[:-1]}……]"
        else:
            return ret
