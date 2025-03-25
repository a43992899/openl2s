import json
import os
import sys


# Generate [audiopath, lrcpath] json file
def get_json_list(audio_dir, vad_dir, json_dir):
    # audio_dir: audio file directory
    # lrc_dir: lrc file directory
    # json_dir: json file save path
    result_list = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(".Vocals.mp3"):
            audio_path = os.path.join(audio_dir, filename)
            vad_path = os.path.join(vad_dir, filename.split(".")[0] + ".txt")
            result_list.append([audio_path, vad_path])

    with open(json_dir, "w") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

audio_dir = sys.argv[1]
vad_dir = sys.argv[2]
json_dir = sys.argv[3]

get_json_list(audio_dir, vad_dir, json_dir)