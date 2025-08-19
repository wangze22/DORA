import pathlib
import re
import os
from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, calculate_activation_statistics
from pytorch_fid.fid_score import compute_statistics_of_path as original_compute_stats


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def compute_statistics_of_files(files, model, batch_size, dims, device, num_workers = 0):
    m, s = calculate_activation_statistics(files, model, batch_size, dims, device, num_workers)
    return m, s


def calculate_fid_given_paths_sliding_window(paths, window_size, batch_size, device, dims, num_workers = 0):
    if not all(os.path.exists(p) for p in paths):
        raise RuntimeError(f'Invalid path in {paths}')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # Compute statistics for the first path (reference)
    m1, s1 = original_compute_stats(paths[0], model, batch_size, dims, device, num_workers)

    # Get and sort all images in the second path
    second_path = pathlib.Path(paths[1])
    all_files = sorted([str(f) for f in second_path.glob('*.png')], key = natural_sort_key)

    fid_scores = []
    num_windows = len(all_files) // window_size  # Calculate how many full windows we can process

    # Process each window
    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size
        current_files = all_files[start_index:end_index]
        m2, s2 = compute_statistics_of_files(current_files, model, batch_size, dims, device, num_workers)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        fid_scores.append(fid_value)

    return fid_scores


def calculate_fid_given_all_paths_sliding_window(paths, names, window_size, batch_size, device, dims, num_workers = 0, baseline_data = None):
    if not all(os.path.exists(p) for p in paths):
        raise RuntimeError("One or more paths are invalid")
    assert len(names) == len(paths)
    # Load the model for FID calculation
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # Compute statistics for the baseline path (reference)
    if baseline_data is None:
        m1, s1 = original_compute_stats(paths[0], model, batch_size, dims, device, num_workers = 1)
    else:
        m1 = baseline_data[0]
        s1 = baseline_data[1]

    # Dictionary to store FID scores for each category
    fid_scores_dict = {}

    # Loop through all remaining paths
    for path, name in zip(paths[1:], names[1:]):  # Start from 1 to skip the baseline
        second_path = pathlib.Path(path)
        all_files = sorted([str(f) for f in second_path.glob('*.png')], key = natural_sort_key)

        fid_scores = []
        window_size = min(len(all_files),window_size)
        num_windows = len(all_files) // window_size  # Calculate how many full windows we can process

        # Process each window
        for i in range(num_windows):
            start_index = i * window_size
            end_index = start_index + window_size
            current_files = all_files[start_index:end_index]
            m2, s2 = compute_statistics_of_files(current_files, model, batch_size, dims, device, num_workers)
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
            fid_scores.append(fid_value)

        fid_scores_dict[name] = fid_scores

    return fid_scores_dict