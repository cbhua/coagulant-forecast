from torch.utils.data import Dataset


class KWaterMLP(Dataset):
    def __init__(self, data) -> None:
        self.data = data # [total number of data, 9]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        '''
        Returns:
            x: <numpy.array> [8] input feature
            y: <numpy.array> [1] coaugulent
        '''
        return self.data[index, :-1], self.data[index, -1]


class KWaterGRU(Dataset):
    def __init__(self, data_list: list, windows_size: int, sliding: int=1) -> None:
        self.data_list = data_list
        self.windows_size = windows_size
        self.sliding = sliding
        self.data_len = [] # 8685, 8241, 8848
        for i in range(len(self.data_list)):
            self.data_len.append(int((self.data_list[i].shape[0] - self.windows_size) / self.sliding) - 1)

    def __len__(self):
        return sum(self.data_len)

    def __getitem__(self, index):
        '''
        Returns:
            x: <torch.tensor> [batch_size, windows_size, feature_size] all 8 features except the lower right coner is 0
            y: <torch.tensor> [batch_size, windows_size, feture_size] reconstruction data
            z: <torch.tensor> [batch_size, feature_size] prediction values
        '''
        flag = 0
        for i in range(len(self.data_len)):
            if index > self.data_len[i]:
                index -= self.data_len[i]
                flag += 1
            else:
                break
        start = index * self.sliding
        end = index * self.sliding + self.windows_size

        x = self.data_list[flag][start:end, :].copy()
        x[-1, -1] = 0
        y = self.data_list[flag][start:end, :].copy()
        z = self.data_list[flag][end, :].copy()
        return x, y, z 


class KWaterOur(Dataset):
    def __init__(self, data_list: list, windows_size: int, sliding: int=1) -> None:
        self.data_list = data_list
        self.windows_size = windows_size
        self.sliding = sliding
        self.data_len = [] # 8685, 8241, 8848
        for i in range(len(self.data_list)):
            self.data_len.append(int((self.data_list[i].shape[0] - self.windows_size) / self.sliding) - 1)

    def __len__(self):
        return sum(self.data_len)

    def __getitem__(self, index):
        flag = 0
        for i in range(len(self.data_len)):
            if index > self.data_len[i]:
                index -= self.data_len[i]
                flag += 1
            else:
                break
        start = index * self.sliding
        end = index * self.sliding + self.windows_size

        x = self.data_list[flag][start:end, :].copy()
        x[-1, -2:] = 0
        y = self.data_list[flag][start:end, :].copy()
        z = self.data_list[flag][end, :].copy()
        return x, y, z 