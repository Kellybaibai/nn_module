
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class Dataset:
    def __init__(self, dataset_name, root=None, train=True, transform=None, target_transform=None, download=True):
        self.dataset_name = dataset_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data = None
        self.targets = None

        if dataset_name == 'iris':
            self.load_iris()
        elif dataset_name == 'mnist':
            self.load_mnist()
        elif dataset_name == 'fashion-mnist':
            self.load_fashion_mnist()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def load_iris(self):
        iris = load_iris()
        self.data = iris.data
        self.targets = iris.target

    def load_mnist(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root=self.root, train=self.train, transform=self.transform,
                                 target_transform=self.target_transform, download=self.download)
        self.data = dataset.data.numpy()
        self.targets = dataset.targets.numpy()

    def load_fashion_mnist(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.FashionMNIST(root=self.root, train=self.train, transform=self.transform,
                                        target_transform=self.target_transform, download=self.download)
        self.data = dataset.data.numpy()
        self.targets = dataset.targets.numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def split_train_test(self, test_size=0.2, random_state=42):
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty. Cannot split.")

        data_size = len(self.dataset)
        indices = list(range(data_size))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

        train_dataset = self._subset_dataset(train_indices)
        test_dataset = self._subset_dataset(test_indices)

        return train_dataset, test_dataset

    def _subset_dataset(self, indices):
        subset = Subset(self.dataset, indices)
        return CustomDataLoader(subset, batch_size=self.batch_size, shuffle=self.shuffle)

