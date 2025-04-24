#!/usr/bin/env python3

import torch
import os
import librosa
import re
import json
from tqdm import tqdm
from functools import partial
import sys
from models.fireredasr_mp import FireRedAsr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

input_dir = sys.argv[1]
output_path = sys.argv[2]
num_jobs = int(sys.argv[3])


device = None
model = None
model_dir = "asr/pretrained_models"

batch_size = 4
device_count = torch.cuda.device_count()
shared_number = None
pattern = r'\[(\d{2}):(\d{2}).(\d{2})\].*?'

def load_model():
    global model
    model = FireRedAsr(device, model_dir, half=True)

def create_segments(wav, timestamps):
    vad_timestamps = [re.findall(pattern, i)[0] for i in timestamps]
    vad_timestamps_sec = [int(i[0]) * 60 + int(i[1]) + float(i[2]) / 100 for i in vad_timestamps]
    vad_sample_points = [int(i) * 16000 for i in vad_timestamps_sec]
    vad_sample_points = vad_sample_points + [wav.shape[-1]]
    segments = [wav[vad_sample_points[i]:vad_sample_points[i + 1]] for i in range(len(vad_sample_points) - 1)]
    return vad_timestamps, segments

def gen_lrc_file(asr_lrcs, lrc_timestamps):
    lrc_content = []
    for timestamp, lrc in zip(lrc_timestamps, asr_lrcs):
        lrc_content.append(f"[{timestamp[0]}:{timestamp[1]}.{timestamp[2]}]{lrc}")
    return lrc_content

import audioread
def main(paths, output_dir):
    global device
    if device is None:
        with shared_number.get_lock():
            device = torch.device(f"cuda:{shared_number.value % device_count}")
            shared_number.value += 1
        load_model()

    try:
        wav_path, vad_path = paths
        name = os.path.basename(wav_path).split(".")[0]
        dec = audioread.ffdec.FFmpegAudioFile(wav_path)
        wav, sr = librosa.load(dec, sr=16000)
        wav = torch.from_numpy(wav).to(device)
        with open(vad_path, "r") as f:
            timestamps = [i.strip() for i in f.readlines()]
        vad_timestamps, segments = create_segments(wav, timestamps)
        
        sample_rate = 16000
        max_segment_times = 30
        custom_segment_sizes = max_segment_times * sample_rate
        
        asr_lrcs = []
        for i in range(len(segments)):
            if len(segments[i]) > custom_segment_sizes:
                segments[i] = segments[i][:custom_segment_sizes]


        for i in range(0, len(segments), batch_size):
            batch_wavs = segments[i:i + batch_size]
            wav_lengths = torch.LongTensor([i.shape[-1] for i in batch_wavs]).to(device)

            results = model.transcribe(
                batch_wavs, wav_lengths,
                {
                    "beam_size": 3,
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.25,
                    "aed_length_penalty": 0.6,
                    "eos_penalty": 1.0,
                    "decode_min_len": 0,
                }
            )
            asr_lrcs.extend(results)

        new_lrcs = gen_lrc_file(asr_lrcs, vad_timestamps)
        with open(os.path.join(output_dir, f"{name}.lrc"), "w") as f:
            f.write("\n".join(new_lrcs))
    except Exception as e:
        print((f"Error in handling wav | {wav_path} | {vad_path} | {e}"))

def init_process(device_number):
    global shared_number
    shared_number = device_number

if __name__ == "__main__":
    data_list = []
    
    with open(input_dir, "r") as f:
        datas = json.load(f)

    os.makedirs(output_path, exist_ok=True)
    
    already_list = os.listdir(output_path)
    already_list = set([Path(name).stem for name in already_list])
    data_list = [(data[0], data[1]) for data in datas if os.path.basename(data[0]).split(".")[0] not in already_list]
    
    print("*** the length to all data ***", len(datas))
    print("*** the length already data ***", len(already_list))
    print("*** the length to handle data ***", len(data_list))
    
    
    device_number = torch.multiprocessing.Value('i', 0)
    wrapper = partial(main, output_dir=output_path)

    with torch.multiprocessing.Pool(num_jobs, initializer=init_process, initargs=(device_number, )) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapper, data_list
            ),
            total=len(data_list)
        ):
            pass
