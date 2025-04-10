import os
import multiprocessing as mp
import argparse
from tqdm import tqdm
import random

os.environ['OMP_NUM_THREADS'] = '10'
os.environ['OMP_INTER_THREADS'] = '2'

def get_input_files(input_path):
    if os.path.isdir(input_path):
        # Get all files from the directory recursively
        input_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                input_files.append(os.path.join(root, file))
        return input_files
    elif os.path.isfile(input_path):
        # Assume it's a filelist
        with open(input_path, 'r') as f:
            input_files = [line.strip() for line in f.readlines()]
        return input_files
    else:
        raise ValueError(f"Input path {input_path} is neither a directory nor a file.")

def slice_filelist(cur_shard, total_shard, filelist):
    total_files = len(filelist)
    start_idx = cur_shard * total_files // total_shard
    end_idx = (cur_shard + 1) * total_files // total_shard
    if cur_shard == total_shard - 1:
        end_idx = total_files
    return filelist[start_idx:end_idx]

def msa(gpu_id, cur_shard, total_shard, input_files, output_dir):
    import allin1  # Assuming allin1 is a module available in your environment

    print(f"Processing shard {cur_shard} using GPU {gpu_id}...")

    # Slice the input files according to the shard
    input_files_slice = slice_filelist(cur_shard, total_shard, input_files)
    print(f"Shard {cur_shard} has {len(input_files_slice)} items")

    # Shuffle the input files
    random.shuffle(input_files_slice)

    # Create output directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    demix_dir = os.path.join(output_dir, 'demix')
    spec_dir = os.path.join(output_dir, 'spec')
    os.makedirs(demix_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)

    # Process the audio files using allin1
    print(f"Analyzing audio files...")
    batch_size = 64
    for i in range(0, len(input_files_slice), batch_size):
        batch = input_files_slice[i:i+batch_size]
        try:
            result = allin1.analyze(
                batch,
                out_dir=output_dir,
                demix_dir=demix_dir,
                spec_dir=spec_dir,
                device=f"cuda:{gpu_id}"
            )
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio files using multiple GPUs.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input directory or filelist')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--gpu_ids', '-g', type=str, default='0,1,2,3,4,5,6,7', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--total_shard', '-t', type=int, default=8, help='Total number of shards, 1 shard per GPU')
    parser.add_argument('--start_from_shard', '-s', type=int, default=0, help='Shard index to start from')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_files = get_input_files(args.input)
    print(f"Total {len(input_files)} input files.")

    processes = []
    gpu_ids = args.gpu_ids.split(',')
    total_shard = args.total_shard
    start_from_shard = args.start_from_shard

    for idx, gpu_id in enumerate(gpu_ids):
        cur_shard = idx + start_from_shard
        process = mp.Process(target=msa, args=(gpu_id, cur_shard, total_shard, input_files, args.output))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("All processes finished.")
