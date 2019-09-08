from torch.utils.data import Dataset


class AggregatorDataset(Dataset):
    def __init__(self, datasets):
	    # TODO: Check type of datasets
        self.datasets = datasets

        self.lens = []
        for ds in self.datasets:
            self.lens.append(len(ds))

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
		# TODO: get rid of for loop
        for i in range(len(self.datasets)):
            if idx < np.sum(self.lens[:(i+1)]):
                return self.datasets[i][idx - sum(self.lens[:i])]