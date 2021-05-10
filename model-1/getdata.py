import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]

class DatasetFromHdf5_ex(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_ex, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')
        self.name = hf.get('name')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float(), self.name[index].decode('utf-8')

    def __len__(self):
        return self.data.shape[0]

class DatasetFromHdf5_frame_1(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_frame_1, self).__init__()
        hf = h5py.File(file_path)
        self.data_1 = hf.get('data_1')
        self.target = hf.get('label_1')
        self.name = hf.get('name')

    def __getitem__(self, index):
        return torch.from_numpy(self.data_1[index,:,:,:]).float(),\
               torch.from_numpy(self.target[index,:,:,:]).float(),\
               self.name[index].decode('utf-8')

    def __len__(self):
        return self.target.shape[0]

class DatasetFromHdf5_frame_3(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_frame_3, self).__init__()
        hf = h5py.File(file_path)
        self.data_1 = hf.get('data_1')
        self.data_2 = hf.get('data_2')
        self.data_3 = hf.get('data_3')
        self.target = hf.get('label')
        self.name = hf.get('name')

    def __getitem__(self, index):
        return torch.from_numpy(self.data_1[index,:,:,:]).float(), \
               torch.from_numpy(self.data_2[index,:,:,:]).float(), \
               torch.from_numpy(self.data_3[index,:,:,:]).float(), \
               torch.from_numpy(self.target[index,:,:,:]).float(), \
               self.name[index].decode('utf-8')

    def __len__(self):
        return self.target.shape[0]

class DatasetFromHdf5_frame_5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_frame_5, self).__init__()
        hf = h5py.File(file_path)
        self.data_1 = hf.get('data_1')
        self.data_2 = hf.get('data_2')
        self.data_3 = hf.get('data_3')
        self.data_4 = hf.get('data_4')
        self.data_5 = hf.get('data_5')
        self.target = hf.get('label')
        self.name = hf.get('name')

    def __getitem__(self, index):
        return torch.from_numpy(self.data_1[index,:,:,:]).float(), \
               torch.from_numpy(self.data_2[index,:,:,:]).float(), \
               torch.from_numpy(self.data_3[index,:,:,:]).float(), \
               torch.from_numpy(self.data_4[index,:,:,:]).float(), \
               torch.from_numpy(self.data_5[index,:,:,:]).float(), \
               torch.from_numpy(self.target[index,:,:,:]).float(),\
               self.name[index].decode('utf-8')

    def __len__(self):
        return self.target.shape[0]