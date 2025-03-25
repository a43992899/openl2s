import webrtcvad
import torch.multiprocessing as mp
import os
import threading
from tqdm import tqdm
import traceback
import argparse
import glob

vocal_file_lock = threading.Lock()
bgm_file_lock = threading.Lock()

from vad_tool import read_wave_to_frames, vad_generator, cut_points_generator, cut_points_storage_generator, wavs_generator

LOGGING_INTERVAL = 3
SAMPLE_RATE = 16000
FRAME_DURATION = 10


MIN_ACTIVE_TIME_MS = 100
SIL_HEAD_TAIL_MS = 50
SIL_MID_MS = 200
CUT_MIN_MS = 800
CUT_MAX_MS = 20000

MIN_ACTIVE_FRAME = MIN_ACTIVE_TIME_MS // FRAME_DURATION
SIL_FRAME = SIL_HEAD_TAIL_MS // FRAME_DURATION
SIL_MID_FRAME = SIL_MID_MS // FRAME_DURATION
CUT_MIN_FRAME = CUT_MIN_MS // FRAME_DURATION
CUT_MAX_FRAME = CUT_MAX_MS // FRAME_DURATION
RANDOM_MIN_FRAME = True

def inference(rank, out_dir, filelist_name, queue: mp.Queue):
    info_dir = os.path.join(out_dir, "vad_info")
    os.makedirs(info_dir, exist_ok=True)
    
    while True:
        input_path = queue.get()
        if input_path is None:
            break
        try:
            vad_tools = webrtcvad.Vad(3) # create a new vad each time to avoid some bugs
            vocal_path = input_path[0]
            filename = os.path.basename(vocal_path).split(".")[0]
            frames, wav = read_wave_to_frames(vocal_path, SAMPLE_RATE, FRAME_DURATION)
            vad_info = vad_generator(frames, SAMPLE_RATE, vad_tools)

            cut_points = cut_points_generator(vad_info, MIN_ACTIVE_FRAME, SIL_FRAME, SIL_MID_FRAME, CUT_MIN_FRAME, CUT_MAX_FRAME, RANDOM_MIN_FRAME)
            raw_vad_content, file_content = cut_points_storage_generator(vad_info, cut_points, CUT_MIN_MS)

            with open(os.path.join(info_dir, filename+".txt"), "w") as f:
                f.write(file_content)


        except Exception as e:
            traceback.print_exc()
            print(e)

def setInterval(interval):
    def decorator(function):
        def wrapper(*args, **kwargs):
            stopped = threading.Event()

            def loop():  # executed in another thread
                while not stopped.wait(interval):  # until stopped
                    function(*args, **kwargs)

            t = threading.Thread(target=loop)
            t.daemon = True  # stop if the program exits
            t.start()
            return stopped

        return wrapper

    return decorator


last_batches = None


@setInterval(LOGGING_INTERVAL)
def QueueWatcher(queue, bar):
    global last_batches
    curr_batches = queue.qsize()
    bar.update(last_batches-curr_batches)
    last_batches = curr_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist_or_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--jobs", type=int, required=False, default=2, help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, required=False, default="large-v3", help="Path to save checkpoints")
    parser.add_argument("--model_dir", type=str, required=False, default="large-v3", help="Path to save checkpoints")
    args = parser.parse_args()

    filelist_or_dir = args.filelist_or_dir
    out_dir = args.out_dir
    NUM_THREADS = args.jobs

    if os.path.isfile(filelist_or_dir):
        filelist_name = filelist_or_dir.split('/')[-1].split('.')[0]
        generator = [x for x in open(filelist_or_dir).read().splitlines()]
    else:
        filelist_name = "vocals"
        generator = [x for x in glob.glob(f"{filelist_or_dir}/*.mp3") if x.endswith(".Vocals.mp3")]
    
    mp.set_start_method('spawn',force=True)

    print(f"Running with {NUM_THREADS} threads and batchsize 1")
    processes = []
    queue = mp.Queue()
    for rank in range(NUM_THREADS):
        p = mp.Process(target=inference, args=(rank, out_dir, filelist_name, queue), daemon=True)
        p.start()
        processes.append(p)

    accum = []
    tmp_file = []
    for filename in tqdm(generator):
        accum.append(filename)
        if len(accum) == 1:
            queue.put(accum.copy())
            accum.clear()

    for _ in range(NUM_THREADS):
        queue.put(None)

    last_batches = queue.qsize()
    bar = tqdm(total=last_batches)
    queue_watcher = QueueWatcher(queue, bar)
    for p in processes:
        p.join()
    queue_watcher.set()