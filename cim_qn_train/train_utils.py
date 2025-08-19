'''
@author: Wang Ze
'''
import torch
from torch.utils.data import DataLoader, Subset


def get_first_n_images(loader, num_imgs, return_type = 'tensor'):
    data_iter = iter(loader)
    images, labels = next(data_iter)

    if num_imgs < images.size(0):
        images = images[:num_imgs]
        labels = labels[:num_imgs]
    else:
        extracted_images = list(images)
        extracted_labels = list(labels)
        while len(extracted_images) < num_imgs:
            batch_images, batch_labels = next(data_iter)
            extracted_images.extend(batch_images)
            extracted_labels.extend(batch_labels)

        images = torch.stack(extracted_images[:num_imgs])
        labels = torch.stack(extracted_labels[:num_imgs])

    if return_type == 'np':
        images = images.numpy()
        labels = labels.numpy()

    return images, labels


def get_subset_loader(dataset, n, batch_size, shuffle = True, num_workers = 0):
    subset_indices = list(range(n))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(dataset = subset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loader


def mvm_time_est_144k(cols, it_time = 2):
    k2 = 3.008e-7
    b2 = 1.083848e-5
    b1 = 2.50952e-5
    T = (k2 * it_time + b2) * cols + b1
    return T


if __name__ == '__main__':
    t = mvm_time_est_144k(64, it_time = 2) * 32768
    print(t)
