from torch.utils.data import Dataset


class Config:
    def __init__(self, root, state, batch, workers):
        self.root = root
        self.state = state
        self.batch = batch
        self.workers = workers


class lmdbDataset(Dataset):
    def __getitem__(self, index):
        pass
