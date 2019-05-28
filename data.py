import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch


def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name=='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


#----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample

def get_skew_indices(dataset, class_to_skew=5, skew_ratio=0.9):
    '''
    Returns a pair of indexes - first one with a class skewed + dataset and other with just the remaining samples of the skewed class
    :param dataset: 
    :param class_to_skew: 
    :param skew_ratio: 
    :return: 
    '''
    class_indices = []
    other_indices = []
    example_samples = {}
    for index in range(len(dataset)):
        if hasattr(dataset, "train_labels"):
            if dataset.target_transform is None:
                label = dataset.train_labels[index]
            else:
                label = dataset.target_transform(dataset.train_labels[index])
        elif hasattr(dataset, "test_labels"):
            if dataset.target_transform is None:
                label = dataset.test_labels[index]
            else:
                label = dataset.target_transform(dataset.test_labels[index])
        else:
            label = dataset[index][1]
        if label == class_to_skew:
            class_indices.append(index)
        else:
            other_indices.append(index)
        if label not in example_samples:
            example_samples[label] = index

    print("Class indices: {}\nOther indices: {}".format(len(class_indices), len(other_indices)))

    stop_idx = int(skew_ratio*len(class_indices))
    skew_class_to_include_first = class_indices[0:stop_idx]
    skew_class_to_include_second = class_indices[stop_idx:]
    first_indices = sorted(other_indices + skew_class_to_include_first)
    second_indices = skew_class_to_include_second# + list(example_samples.values())  # Include one example of each class just to make this code work

    print("first_indices: {}\nsecond_indices: {}".format(len(first_indices), len(second_indices)))

    return first_indices, second_indices


class IndexSubsampledDataset(Dataset):
    '''
    Returns a dataset with only the sub_indexes in the dataset
    '''

    def __init__(self, original_dataset, sub_indices):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indices = sub_indices

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        return sample

class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


#----------------------------------------------------------------------------------------------------------#


# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
}


#----------------------------------------------------------------------------------------------------------#


def get_multitask_experiment(name, scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                             exception=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # prepare datasets
            train_datasets = []
            test_datasets = []
            for task_id, p in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(get_dataset('mnist', type="train", permutation=p, dir=data_dir,
                                                  target_transform=target_transform, verbose=verbose))
                test_datasets.append(get_dataset('mnist', type="test", permutation=p, dir=data_dir,
                                                 target_transform=target_transform, verbose=verbose))
    elif name == 'splitMNIST':
        # check for number of tasks
        assert tasks == 2, "Can have only two tasks!"
        classes_per_task = 10    #TODO This should be 10 but fix later
        # configurations
        config = DATASET_CONFIGS['mnist28']
        if not only_config:
            # prepare train and test datasets with all classes
            target_transform = transforms.Lambda(lambda y, x=None: y)
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform, verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform, verbose=verbose)

            # split them up into sub-tasks - one class is imbalanced across the two tasks.
            CLASS_TO_IMBALANCE = 5

            first_train_idxs, second_train_idxs = get_skew_indices(mnist_train, class_to_skew=CLASS_TO_IMBALANCE, skew_ratio=0.05)

            train_datasets = [IndexSubsampledDataset(mnist_train, first_train_idxs), IndexSubsampledDataset(mnist_train, second_train_idxs)]
            test_datasets = [mnist_test, mnist_test]

    elif name == 'splitMNIST_original':
        # check for number of tasks
        if tasks > 10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks
    config['classes'] = 10 if name == 'splitMNIST' else config['classes']
    print("USING {} CLASSES.".format(config['classes']))

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)