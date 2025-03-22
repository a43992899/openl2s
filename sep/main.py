import time
import random
import hashlib
import json
import os
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['OMP_INTER_THREADS'] = '2'
import concurrent.futures
import shutil
import copy
from tqdm import tqdm
import torch
from collections import Counter
from lib_v5 import spec_utils
from pathlib  import Path
from separate import (
    SeperateDemucs, SeperateMDX,  # Model-related
    save_format, prepare_mix_gpu,  # Utility functions
    cuda_available, mps_available, #directml_available,
)
from typing import List
import onnx
import re
import sys
import yaml
from ml_collections import ConfigDict
from collections import Counter

from gui_data.constants import *
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps



is_gpu_available = cuda_available or mps_available# or directml_available

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')

#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')
DOWNLOAD_MODEL_CACHE = os.path.join(BASE_PATH, 'gui_data', 'model_manual_download.json')

DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')
file_check(os.path.join(MODELS_DIR, 'Main_Models'), VR_MODELS_DIR)

model_hash_table = {}


def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''
    with open(dictionary, 'r') as d:
        return json.load(d)

class MainWindow():
    def __init__(self, inputPaths, export_path, main_input_path, use_gpu=True, cuda_idx=0, save_format=FLAC, resume=False, ensemble_type_var=0, read_from_tos=False):
        self.export_path_var = export_path
        self.inputPaths = inputPaths
        if main_input_path.startswith('tos://'):
            self.main_input_path = main_input_path
        elif os.path.isdir(main_input_path):
            self.main_input_path = main_input_path
        else:
            raise Exception("please provide main_input_path, will be used for computing intermediate_folders.")
        if not main_input_path.endswith('/'):
            self.main_input_path = f'{main_input_path}/'
        self.read_from_tos = read_from_tos
        self.mdx_net_model_var = DEFAULT_DATA['mdx_net_model']
        self.demucs_model_var = DEFAULT_DATA['demucs_model']
        self.model_sample_mode_var = DEFAULT_DATA['model_sample_mode']
        self.chosen_process_method_var = ENSEMBLE_MODE
        self.is_testing_audio_var = DEFAULT_DATA['is_testing_audio']
        self.model_sample_mode_duration_var = DEFAULT_DATA['model_sample_mode_duration']
        self.is_add_model_name_var = DEFAULT_DATA['is_add_model_name']
        self.is_create_model_folder_var = DEFAULT_DATA['is_create_model_folder']
        self.ensemble_main_stem_var = VOCAL_PAIR
        self.is_secondary_stem_only_var = DEFAULT_DATA['is_secondary_stem_only']
        self.is_primary_stem_only_var = DEFAULT_DATA['is_primary_stem_only']
        self.set_vocal_splitter_var = DEFAULT_DATA['set_vocal_splitter']
        self.is_set_vocal_splitter_var = DEFAULT_DATA['is_set_vocal_splitter']
        self.is_task_complete_var = DEFAULT_DATA['is_task_complete']
        self.ensemble_type_var = MIN_MIX if ensemble_type_var == 0 else MAX_MIN
        self.device_set_var = str(cuda_idx)
        self.is_deverb_vocals_var = DEFAULT_DATA['is_deverb_vocals']
        self.deverb_vocal_opt_var = DEFAULT_DATA['deverb_vocal_opt']
        self.denoise_option_var = DEFAULT_DATA['denoise_option']
        self.is_gpu_conversion_var = use_gpu
        self.is_normalization_var = DEFAULT_DATA['is_normalization']
        self.is_mdx_c_seg_def_var = DEFAULT_DATA['is_mdx_c_seg_def']
        self.mdx_batch_size_var = DEFAULT_DATA['mdx_batch_size']
        self.mdx_stems_var = DEFAULT_DATA['mdx_stems']
        self.overlap_var = DEFAULT_DATA['overlap']
        self.overlap_mdx_var = DEFAULT_DATA['overlap_mdx']
        self.overlap_mdx23_var = DEFAULT_DATA['overlap_mdx23']
        self.semitone_shift_var = DEFAULT_DATA['semitone_shift']
        self.is_match_frequency_pitch_var = DEFAULT_DATA['is_match_frequency_pitch']
        self.is_mdx23_combine_stems_var = DEFAULT_DATA['is_mdx23_combine_stems']
        self.mp3_bit_set_var = DEFAULT_DATA['mp3_bit_set']
        self.save_format_var = save_format
        self.is_invert_spec_var = DEFAULT_DATA['is_invert_spec']
        self.demucs_stems_var = DEFAULT_DATA['demucs_stems']
        self.is_demucs_combine_stems_var = DEFAULT_DATA['is_demucs_combine_stems']
        self.is_save_inst_set_vocal_splitter_var = DEFAULT_DATA['is_save_inst_set_vocal_splitter']
        self.is_demucs_pre_proc_model_activate_var = DEFAULT_DATA['is_demucs_pre_proc_model_activate']
        self.mdx_is_secondary_model_activate_var = DEFAULT_DATA['mdx_is_secondary_model_activate']
        self.margin_var = DEFAULT_DATA['margin']
        self.mdx_segment_size_var = DEFAULT_DATA['mdx_segment_size']
        self.compensate_var = DEFAULT_DATA['compensate']
        self.demucs_is_secondary_model_activate_var = DEFAULT_DATA['demucs_is_secondary_model_activate']
        self.margin_demucs_var = DEFAULT_DATA['margin_demucs']
        self.shifts_var = DEFAULT_DATA['shifts']
        self.segment_var = DEFAULT_DATA['segment']
        self.is_split_mode_var = DEFAULT_DATA['is_split_mode']
        self.is_chunk_demucs_var = DEFAULT_DATA['is_chunk_demucs']
        self.is_primary_stem_only_Demucs_var = DEFAULT_DATA['is_primary_stem_only_Demucs']
        self.is_secondary_stem_only_Demucs_var = DEFAULT_DATA['is_secondary_stem_only_Demucs']
        self.is_demucs_pre_proc_model_inst_mix_var = DEFAULT_DATA['is_demucs_pre_proc_model_inst_mix']
        self.is_save_all_outputs_ensemble_var = DEFAULT_DATA['is_save_all_outputs_ensemble']
        self.chosen_ensemble_var = CHOOSE_ENSEMBLE_OPTION
        self.is_append_ensemble_name_var = DEFAULT_DATA['is_append_ensemble_name']
        self.is_wav_ensemble_var = DEFAULT_DATA['is_wav_ensemble']
        self.time_window_var = DEFAULT_DATA['time_window']
        self.choose_algorithm_var = DEFAULT_DATA['choose_algorithm']
        self.intro_analysis_var = DEFAULT_DATA['intro_analysis']
        self.db_analysis_var = DEFAULT_DATA['db_analysis']
        self.is_save_align_var = DEFAULT_DATA['is_save_align']
        self.is_match_silence_var = DEFAULT_DATA['is_match_silence']
        self.is_spec_match_var = DEFAULT_DATA['is_spec_match']
        self.phase_option_var = DEFAULT_DATA['phase_option']
        self.phase_shifts_var = DEFAULT_DATA['phase_shifts']
        self.time_stretch_rate_var = DEFAULT_DATA['time_stretch_rate']
        self.pitch_rate_var = DEFAULT_DATA['pitch_rate']
        self.is_time_correction_var = DEFAULT_DATA['is_time_correction']
        self.mdxnet_stems_var = ALL_STEMS
        self.wav_type_set = None
        self.is_primary_stem_only_Demucs_Text_var = ''
        self.is_secondary_stem_only_Demucs_Text_var = ''
        self.is_secondary_stem_only_Text_var = "Instrumental Only"
        self.is_primary_stem_only_Text_var = "Vocals Only"
        self.vr_hash_MAPPER = load_model_hash_data(VR_HASH_JSON)
        self.mdx_hash_MAPPER = load_model_hash_data(MDX_HASH_JSON)
        self.mdx_name_select_MAPPER = load_model_hash_data(MDX_MODEL_NAME_SELECT)
        self.demucs_name_select_MAPPER = load_model_hash_data(DEMUCS_MODEL_NAME_SELECT)
        self.resume = resume

        


    def model_list(self, primary_stem: str, secondary_stem: str, is_4_stem_check=False, is_multi_stem=False, is_dry_check=False, is_no_demucs=False, is_check_vocal_split=False):
            
        stem_check = self.assemble_model_data(arch_type=ENSEMBLE_STEM_CHECK, is_dry_check=is_dry_check)
        
        def matches_stem(model: ModelData):
            primary_match = model.primary_stem in {primary_stem, secondary_stem}
            mdx_stem_match = primary_stem in model.mdx_model_stems and model.mdx_stem_count <= 2
            return primary_match or mdx_stem_match if is_no_demucs else primary_match or primary_stem in model.mdx_model_stems

        result = []

        for model in stem_check:
            if is_multi_stem:
                result.append(model.model_and_process_tag)
            elif is_4_stem_check and (model.demucs_stem_count == 4 or model.mdx_stem_count == 4):
                result.append(model.model_and_process_tag)
            elif matches_stem(model) or (not is_no_demucs and primary_stem.lower() in model.demucs_source_list):
                if is_check_vocal_split:
                    model_name = None if model.is_karaoke or not model.vocal_split_model else model.model_basename
                else: 
                    model_name = model.model_and_process_tag
                    
                result.append(model_name)

        return result

    def ensemble_listbox_get_all_selected_models(self):
        return [
                # "VR Arc: UVR-DeNoise-Lite", 
                "MDX-Net: Kim Vocal 1", 
                "MDX-Net: UVR-MDX-NET Inst 3", 
                "Demucs: v4 | htdemucs_ft"]

    def assemble_model_data(self, model=None, arch_type=ENSEMBLE_MODE, is_dry_check=False, is_change_def=False, is_get_hash_dir_only=False):

        if arch_type == ENSEMBLE_STEM_CHECK:
            
            model_data = self.model_data_table
            missing_models = [model.model_status for model in model_data if not model.model_status]
            
            if missing_models or not model_data:
                model_data: List[ModelData] = [ModelData(model_name, is_dry_check=is_dry_check) for model_name in self.ensemble_model_list]
                self.model_data_table = model_data

        if arch_type == ENSEMBLE_MODE:
            model_data: List[ModelData] = [ModelData(model_name) for model_name in self.ensemble_listbox_get_all_selected_models()]
        if arch_type == ENSEMBLE_CHECK:
            model_data: List[ModelData] = [ModelData(model, is_change_def=is_change_def, is_get_hash_dir_only=is_get_hash_dir_only)]
        if arch_type == MDX_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, MDX_ARCH_TYPE)]
        if arch_type == DEMUCS_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, DEMUCS_ARCH_TYPE)]#

        return model_data

    def cached_source_model_list_check(self, model_list): # model_list: List[ModelData]

        model: ModelData
        primary_model_names = lambda process_method:[model.model_basename if model.process_method == process_method else None for model in model_list]
        secondary_model_names = lambda process_method:[model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == process_method else None for model in model_list]

        self.mdx_primary_model_names = primary_model_names(MDX_ARCH_TYPE)
        self.demucs_primary_model_names = primary_model_names(DEMUCS_ARCH_TYPE)
        self.mdx_secondary_model_names = secondary_model_names(MDX_ARCH_TYPE)
        self.demucs_secondary_model_names = [model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == DEMUCS_ARCH_TYPE and not model.secondary_model is None else None for model in model_list]
        self.demucs_pre_proc_model_name = [model.pre_proc_model.model_basename if model.pre_proc_model else None for model in model_list]#list(dict.fromkeys())
        
        for model in model_list:
            if model.process_method == DEMUCS_ARCH_TYPE and model.is_demucs_4_stem_secondaries:
                if not model.is_4_stem_ensemble:
                    self.demucs_secondary_model_names = model.secondary_model_4_stem_model_names_list
                    break
                else:
                    for i in model.secondary_model_4_stem_model_names_list:
                        self.demucs_secondary_model_names.append(i)
        
        self.all_models = self.mdx_primary_model_names + self.demucs_primary_model_names + self.mdx_secondary_model_names + self.demucs_secondary_model_names + self.demucs_pre_proc_model_name
    
    def determine_voc_split(self, models):
        is_vocal_active = self.check_only_selection_stem(VOCAL_STEM_ONLY) or self.check_only_selection_stem(INST_STEM_ONLY)

        if self.set_vocal_splitter_var != NO_MODEL and self.is_set_vocal_splitter_var and is_vocal_active:
            model_stems_list = self.model_list(VOCAL_STEM, INST_STEM, is_dry_check=True, is_check_vocal_split=True)
            if any(model.model_basename in model_stems_list for model in models):
                return 1
        
        return 0

    def cached_sources_clear(self):

        self.vr_cache_source_mapper = {}
        self.mdx_cache_source_mapper = {}
        self.demucs_cache_source_mapper = {}

    def process_get_baseText(self, total_files, file_num, is_dual=False):
        """Create the base text for the command widget"""
        
        init_text = 'Files' if is_dual else 'File'
        
        text = '{init_text} {file_num}/{total_files} '.format(init_text=init_text,
                                                              file_num=file_num,
                                                              total_files=total_files)
        
        return text

    def process_update_progress(self, total_files, step: float = 1):
        """Calculate the progress for the progress widget in the GUI"""
        if DISABLE_LOGGING: return
        total_count = self.true_model_count * total_files
        base = (100 / total_count)
        progress = base * self.iteration - base
        progress += base * step

        self.progress_bar_main_var = progress
        
        print(f'Process Progress: {int(progress)}%')

    def check_only_selection_stem(self, checktype):
        
        chosen_method = self.chosen_process_method_var
        is_demucs = chosen_method == DEMUCS_ARCH_TYPE#

        stem_primary_label = self.is_primary_stem_only_Demucs_Text_var if is_demucs else self.is_primary_stem_only_Text_var
        stem_primary_bool = self.is_primary_stem_only_Demucs_var if is_demucs else self.is_primary_stem_only_var
        stem_secondary_label = self.is_secondary_stem_only_Demucs_Text_var if is_demucs else self.is_secondary_stem_only_Text_var
        stem_secondary_bool = self.is_secondary_stem_only_Demucs_var if is_demucs else self.is_secondary_stem_only_var

        if checktype == VOCAL_STEM_ONLY:
            return not (
                (not VOCAL_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (not VOCAL_STEM_ONLY in stem_secondary_label and stem_secondary_bool)
            )
        elif checktype == INST_STEM_ONLY:
            return (
                (INST_STEM_ONLY == stem_primary_label and stem_primary_bool and self.is_save_inst_set_vocal_splitter_var and self.set_vocal_splitter_var != NO_MODEL) or 
                (INST_STEM_ONLY == stem_secondary_label and stem_secondary_bool and self.is_save_inst_set_vocal_splitter_var and self.set_vocal_splitter_var != NO_MODEL)
            )
        elif checktype == IS_SAVE_VOC_ONLY:
            return (
                (VOCAL_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (VOCAL_STEM_ONLY == stem_secondary_label and stem_secondary_bool)
            )
        elif checktype == IS_SAVE_INST_ONLY:
            return (
                (INST_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (INST_STEM_ONLY == stem_secondary_label and stem_secondary_bool)
            )
    
    def return_ensemble_stems(self, is_primary=False): 
        """Grabs and returns the chosen ensemble stems."""
        
        ensemble_stem = self.ensemble_main_stem_var.partition("/")
        
        if is_primary:
            return ensemble_stem[0]
        else:
            return ensemble_stem[0], ensemble_stem[2]
    
    def process_determine_vocal_split_model(self):
        """Obtains the correct vocal splitter secondary model data for conversion."""
        
        # Check if a vocal splitter model is set and if it's not the 'NO_MODEL' value
        if self.set_vocal_splitter_var != NO_MODEL and self.is_set_vocal_splitter_var:
            vocal_splitter_model = ModelData(self.set_vocal_splitter_var, is_vocal_split_model=True)
            
            # Return the model if it's valid
            if vocal_splitter_model.model_status:
                return vocal_splitter_model
                
        return None

    def process_iteration(self):
        self.iteration = self.iteration + 1

    def cached_source_callback(self, process_method, model_name=None):
        
        model, sources = None, None
        
        if process_method == VR_ARCH_TYPE:
            mapper = self.vr_cache_source_mapper
        if process_method == MDX_ARCH_TYPE:
            mapper = self.mdx_cache_source_mapper
        if process_method == DEMUCS_ARCH_TYPE:
            mapper = self.demucs_cache_source_mapper
        
        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value
        
        return model, sources

    def cached_model_source_holder(self, process_method, sources, model_name=None):
        
        if process_method == VR_ARCH_TYPE:
            self.vr_cache_source_mapper = {**self.vr_cache_source_mapper, **{model_name: sources}}
        if process_method == MDX_ARCH_TYPE:
            self.mdx_cache_source_mapper = {**self.mdx_cache_source_mapper, **{model_name: sources}}
        if process_method == DEMUCS_ARCH_TYPE:
            self.demucs_cache_source_mapper = {**self.demucs_cache_source_mapper, **{model_name: sources}}

    def process_start(self):
        """Start the conversion for all the given mp3 and wav files"""
        
        stime = time.perf_counter()
        time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'
        export_path = self.export_path_var
        is_ensemble = False
        self.true_model_count = 0
        self.iteration = 0
        is_verified_audio = True
        inputPaths = self.inputPaths
        inputPath_total_len = len(inputPaths)
        
        
        if self.chosen_process_method_var == ENSEMBLE_MODE:
            model, ensemble = self.assemble_model_data(), Ensembler()
            export_path, is_ensemble = ensemble.ensemble_folder_name, True
        if self.chosen_process_method_var == MDX_ARCH_TYPE:
            model = self.assemble_model_data(self.mdx_net_model_var, MDX_ARCH_TYPE)
        if self.chosen_process_method_var == DEMUCS_ARCH_TYPE:
            model = self.assemble_model_data(self.demucs_model_var, DEMUCS_ARCH_TYPE)

        self.cached_source_model_list_check(model)
        
        true_model_4_stem_count = sum(m.demucs_4_stem_added_count if m.process_method == DEMUCS_ARCH_TYPE else 0 for m in model)
        true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
        self.true_model_count = sum(2 if m.is_secondary_model_activated else 1 for m in model) + true_model_4_stem_count + true_model_pre_proc_model_count + self.determine_voc_split(model)

        model_basename2weight = dict() # cache for seperator instances
        for file_num, audio_file in enumerate(tqdm(inputPaths), start=1):
            try:
                self.cached_sources_clear()

                if USE_IN_MEMORY_FS_TO_CACHE_INTERMEDIATE_RESULTS:
                    in_memory_fs.clear()
                    if not DISABLE_LOGGING: print("In-Memory FS Cleared")

                base_text = self.process_get_baseText(total_files=inputPath_total_len, file_num=file_num)

                if not DISABLE_LOGGING: print(f'{NEW_LINE if not file_num ==1 else NO_LINE}{base_text}"{os.path.basename(audio_file)}\".{NEW_LINES}')
                
                device = torch.device(f'cuda:{self.device_set_var}') if is_gpu_available else torch.device('cpu')

                global tos_client
                if not self.read_from_tos: tos_client = None

                # skip if exists
                intermediate_folders = ''
                if ensemble.main_input_path is not None:
                    prefix_replaced = audio_file.replace(str(ensemble.main_input_path), '')
                    intermediate_folders = os.path.split(prefix_replaced)[0]
                    if intermediate_folders.startswith('/'):
                        intermediate_folders = intermediate_folders[1:]
                    os.makedirs(os.path.join(ensemble.main_export_path, intermediate_folders), exist_ok=True)

                SKIP_FLG = False
                if self.resume:
                    _audio_file_base = os.path.splitext(os.path.basename(audio_file))[0]
                    vocal_path = os.path.join(ensemble.main_export_path, intermediate_folders, f"{_audio_file_base}.Vocals.{self.save_format_var.lower()}")
                    instrumental_path = os.path.join(ensemble.main_export_path, intermediate_folders, f"{_audio_file_base}.Instrumental.{self.save_format_var.lower()}")
                    is_vocal_stem_exist, is_inst_stem_exist = os.path.isfile(vocal_path), os.path.isfile(instrumental_path)
                    SKIP_FLG = is_vocal_stem_exist and is_inst_stem_exist
                    if SKIP_FLG:
                        if not DISABLE_LOGGING: print(f"Skipping {audio_file} as it already exists.")
                        continue
                    
                preload_mix = prepare_mix_gpu(audio_file, device, tos_client=tos_client)

                for current_model_num, current_model in enumerate(model, start=1):
                    self.iteration += 1

                    if is_ensemble:
                        if not DISABLE_LOGGING: print(f'Ensemble Mode - {current_model.model_basename} - Model {current_model_num}/{len(model)}{NEW_LINES}')

                    model_name_text = f'({current_model.model_basename})' if not is_ensemble else ''
                    if not DISABLE_LOGGING: print(base_text + f'{LOADING_MODEL_TEXT} {model_name_text}...')

                    set_progress_bar = lambda step, inference_iterations=0:self.process_update_progress(total_files=inputPath_total_len, step=(step + (inference_iterations)))

                    if not DISABLE_LOGGING: 
                        write_to_console = lambda progress_text, base_text=base_text:print(base_text + progress_text)
                    else:
                        write_to_console = lambda progress_text, base_text=base_text:None

                    audio_file_base = f"{os.path.splitext(os.path.basename(audio_file))[0]}"
                    audio_file_base = audio_file_base if not self.is_testing_audio_var or is_ensemble else f"{round(time.time())}_{audio_file_base}"
                    audio_file_base = audio_file_base if not is_ensemble else f"{audio_file_base}_{current_model.model_basename}"
                    if not is_ensemble:
                        audio_file_base = audio_file_base if not self.is_add_model_name_var else f"{audio_file_base}_{current_model.model_basename}"

                    if self.is_create_model_folder_var and not is_ensemble:
                        export_path = os.path.join(self.export_path_var, current_model.model_basename, os.path.splitext(os.path.basename(audio_file))[0])
                        if not os.path.isdir(export_path):os.makedirs(export_path) 

                    process_data = {
                                    'model_data': current_model, 
                                    'export_path': export_path,
                                    'audio_file_base': audio_file_base,
                                    'audio_file': audio_file,
                                    'set_progress_bar': set_progress_bar,
                                    'write_to_console': write_to_console,
                                    'process_iteration': self.process_iteration,
                                    'cached_source_callback': self.cached_source_callback,
                                    'cached_model_source_holder': self.cached_model_source_holder,
                                    'list_all_models': self.all_models,
                                    'is_ensemble_master': is_ensemble,
                                    'is_4_stem_ensemble': True if self.ensemble_main_stem_var in [FOUR_STEM_ENSEMBLE, MULTI_STEM_ENSEMBLE] and is_ensemble else False}
                    
                    # do audio seperation
                    current_model_basename = current_model.model_basename
                    weight = model_basename2weight.get(current_model_basename, None)

                    if current_model.process_method == MDX_ARCH_TYPE:
                        seperator = SeperateMDX(current_model, process_data)
                        if weight is None:
                            seperator.load_model()
                            weight = copy.deepcopy(seperator.model_run)
                            model_basename2weight[current_model_basename] = weight
                        seperator.model_run = weight

                    if current_model.process_method == DEMUCS_ARCH_TYPE:
                        seperator = SeperateDemucs(current_model, process_data)
                        if weight is None:
                            seperator.load_model()
                            weight = copy.deepcopy(seperator.demucs)
                            model_basename2weight[current_model_basename] = weight
                        seperator.demucs = weight # only support demucs v3 v4

                    seperator.seperate(preload_mix=preload_mix)
                    
                    if is_ensemble:
                        if not DISABLE_LOGGING: print('\n')

                if is_ensemble:
                    if SKIP_FLG: continue
                    audio_file_base = audio_file_base.replace(f"_{current_model.model_basename}","")
                    if not DISABLE_LOGGING: print(base_text + ENSEMBLING_OUTPUTS)
                    
                    if self.ensemble_main_stem_var in [FOUR_STEM_ENSEMBLE, MULTI_STEM_ENSEMBLE]:
                        stem_list = extract_stems(audio_file_base, export_path)
                        for output_stem in stem_list:
                            ensemble.ensemble_outputs(audio_file_base, export_path, output_stem, is_4_stem=True, intermediate_folders=intermediate_folders)
                    else:
                        if not self.is_secondary_stem_only_var:
                            ensemble.ensemble_outputs(audio_file_base, export_path, PRIMARY_STEM, intermediate_folders=intermediate_folders)
                        if not self.is_primary_stem_only_var:
                            ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM, intermediate_folders=intermediate_folders)
                            ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM, is_inst_mix=True, intermediate_folders=intermediate_folders)

                    if not DISABLE_LOGGING: print(DONE)
            except Exception as e:
                print(f'Processing file: {audio_file} failed with error: {e}')    
                
        if not USE_IN_MEMORY_FS_TO_CACHE_INTERMEDIATE_RESULTS:
            shutil.rmtree(export_path) if is_ensemble and len(os.listdir(export_path)) == 0 else None

        # set_progress_bar(1.0)
        if not DISABLE_LOGGING: print(PROCESS_COMPLETE)
        if not DISABLE_LOGGING: print(time_elapsed())
            
        if not DISABLE_LOGGING: print(f'\n{DONE}')



def extract_stems(audio_file_base, export_path):
    
    filenames = [file for file in os.listdir(export_path) if file.startswith(audio_file_base)]

    pattern = r'\(([^()]+)\)(?=[^()]*\.wav)'
    stem_list = []

    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            stem_list.append(match.group(1))
            
    counter = Counter(stem_list)
    filtered_lst = [item for item in stem_list if counter[item] > 1]

    return list(set(filtered_lst))


class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False,
                 is_change_def=False,
                 is_get_hash_dir_only=False,
                 is_vocal_split_model=False):

        device_set = root.device_set_var
        self.DENOISER_MODEL = DENOISER_MODEL_PATH
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH
        self.is_deverb_vocals = root.is_deverb_vocals_var if os.path.isfile(DEVERBER_MODEL_PATH) else False
        self.deverb_vocal_opt = DEVERB_MAPPER[root.deverb_vocal_opt_var]
        self.is_denoise_model = True if root.denoise_option_var == DENOISE_M and os.path.isfile(DENOISER_MODEL_PATH) else False
        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var else -1
        self.is_normalization = root.is_normalization_var#
        self.is_use_opencl = False#True if is_opencl_only else root.is_use_opencl_var
        self.is_primary_stem_only = root.is_primary_stem_only_var
        self.is_secondary_stem_only = root.is_secondary_stem_only_var
        self.is_denoise = True if not root.denoise_option_var == DENOISE_NONE else False
        self.is_mdx_c_seg_def = root.is_mdx_c_seg_def_var#
        self.mdx_batch_size = 1 if root.mdx_batch_size_var == DEF_OPT else int(root.mdx_batch_size_var)
        self.mdxnet_stem_select = root.mdxnet_stems_var
        self.overlap = float(root.overlap_var) if not root.overlap_var == DEFAULT else 0.25
        self.overlap_mdx = float(root.overlap_mdx_var) if not root.overlap_mdx_var == DEFAULT else root.overlap_mdx_var
        self.overlap_mdx23 = int(float(root.overlap_mdx23_var))
        self.semitone_shift = float(root.semitone_shift_var)
        self.is_pitch_change = False if self.semitone_shift == 0 else True
        self.is_match_frequency_pitch = root.is_match_frequency_pitch_var
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = root.is_mdx23_combine_stems_var#
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = None
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = root.wav_type_set#
        self.device_set = device_set.split(':')[-1].strip() if ':' in device_set else device_set
        self.mp3_bit_set = root.mp3_bit_set_var
        self.save_format = root.save_format_var
        self.is_invert_spec = root.is_invert_spec_var#
        self.is_mixer_mode = False#
        self.demucs_stems = root.demucs_stems_var
        self.is_demucs_combine_stems = root.is_demucs_combine_stems_var
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_change_def = is_change_def
        self.model_hash_dir = None
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = root.is_save_inst_set_vocal_splitter_var
        self.is_inst_only_voc_splitter = root.check_only_selection_stem(INST_STEM_ONLY)
        self.is_save_vocal_only = root.check_only_selection_stem(IS_SAVE_VOC_ONLY)

        if selected_process_method == ENSEMBLE_MODE:
            self.process_method, _, self.model_name = model_name.partition(ENSEMBLE_PARTITION)
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = root.return_ensemble_stems()
            
            is_not_secondary_or_pre_proc = not is_secondary_model and not is_pre_proc_model
            self.is_ensemble_mode = is_not_secondary_or_pre_proc
            
            if root.ensemble_main_stem_var == FOUR_STEM_ENSEMBLE:
                self.is_4_stem_ensemble = self.is_ensemble_mode
            elif root.ensemble_main_stem_var == MULTI_STEM_ENSEMBLE and root.chosen_process_method_var == ENSEMBLE_MODE:
                self.is_multi_stem_ensemble = True

            is_not_vocal_stem = self.ensemble_primary_stem != VOCAL_STEM
            self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var if is_not_vocal_stem else False

        if self.process_method == VR_ARCH_TYPE:
            raise NotImplementedError
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = root.mdx_is_secondary_model_activate_var if not is_secondary_model else False
            self.margin = int(root.margin_var)
            self.chunks = 0
            self.mdx_segment_size = int(root.mdx_segment_size_var)
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(MDX_HASH_DIR, root.mdx_hash_MAPPER)
                if self.model_data:
                    
                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

                            self.mdx_c_configs = config
                                
                            if self.mdx_c_configs.training.target_instrument:
                                # Use target_instrument as the primary stem and set 4-stem ensemble to False
                                target = self.mdx_c_configs.training.target_instrument
                                self.mdx_model_stems = [target]
                                self.primary_stem = target
                            else:
                                # If no specific target_instrument, use all instruments in the training config
                                self.mdx_model_stems = self.mdx_c_configs.training.instruments
                                self.mdx_stem_count = len(self.mdx_model_stems)
                                
                                # Set primary stem based on stem count
                                if self.mdx_stem_count == 2:
                                    self.primary_stem = self.mdx_model_stems[0]
                                else:
                                    self.primary_stem = self.mdxnet_stem_select
                                
                                # Update mdxnet_stem_select based on ensemble mode
                                if self.is_ensemble_mode:
                                    self.mdxnet_stem_select = self.ensemble_primary_stem
                        else:
                            self.model_status = False
                    else:
                        self.compensate = self.model_data["compensate"] if root.compensate_var == AUTO_SELECT else float(root.compensate_var)
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()
                        
                    self.secondary_stem = secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = root.demucs_is_secondary_model_activate_var if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var if not root.demucs_stems_var in [VOCAL_STEM, INST_STEM] else False
            self.margin_demucs = int(root.margin_demucs_var)
            self.chunks_demucs = 0
            self.shifts = int(root.shifts_var)
            self.is_split_mode = root.is_split_mode_var
            self.segment = root.segment_var
            self.is_chunk_demucs = root.is_chunk_demucs_var
            self.is_primary_stem_only = root.is_primary_stem_only_var if self.is_ensemble_mode else root.is_primary_stem_only_Demucs_var 
            self.is_secondary_stem_only = root.is_secondary_stem_only_var if self.is_ensemble_mode else root.is_secondary_stem_only_Demucs_var
            self.get_demucs_model_data()
            self.get_demucs_model_path()
            
        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None
            
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        is_secondary_activated_and_status = self.is_secondary_model_activated and self.model_status
        is_demucs = self.process_method == DEMUCS_ARCH_TYPE
        is_all_stems = root.demucs_stems_var == ALL_STEMS
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if is_valid_ensemble or self.is_4_stem_ensemble or is_multi_stem_ensemble_demucs:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = any(i is not None for i in self.secondary_model_4_stem)
                self.demucs_4_stem_added_count -= 1 if self.is_secondary_model_activated else 0
                
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [i.model_basename if i is not None else None for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and is_demucs else self.primary_stem
                self.secondary_model_data(primary_stem)

        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.pre_proc_model_activated = True if self.pre_proc_model else False
                self.is_demucs_pre_proc_model_inst_mix = root.is_demucs_pre_proc_model_inst_mix_var if self.pre_proc_model else False

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = BV_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else LEAD_VOCAL_STEM
            else:
                primary = LEAD_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else BV_VOCAL_STEM
            self.primary_stem, self.secondary_stem = primary, secondary_stem(primary)
            
        self.vocal_splitter_model_data()
            
    def vocal_splitter_model_data(self):
        if not self.is_secondary_model and self.model_status:
            self.vocal_split_model = root.process_determine_vocal_split_model()
            self.is_vocal_split_model_activated = True if self.vocal_split_model else False
            
            if self.vocal_split_model:
                if self.vocal_split_model.bv_model_rebalance:
                    self.is_sec_bv_rebalance = True
            
    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
            
        #print("self.is_secondary_model_activated: ", self.is_secondary_model_activated)
              
    def check_if_karaokee_model(self):
        if IS_KARAOKEE in self.model_data.keys():
            self.is_karaoke = self.model_data[IS_KARAOKEE]
        if IS_BV_MODEL in self.model_data.keys():
            self.is_bv_model = self.model_data[IS_BV_MODEL]#
        if IS_BV_MODEL_REBAL in self.model_data.keys() and self.is_bv_model:
            self.bv_model_rebalance = self.model_data[IS_BV_MODEL_REBAL]#
   
    def get_mdx_model_path(self):
        
        if self.model_name.endswith(CKPT):
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in root.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith(CKPT):
                    ext = ''
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = self.demucs_version in {DEMUCS_V3, DEMUCS_V4}
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in root.demucs_name_select_MAPPER.items():
            if self.model_name == chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        if DEMUCS_UVR_MODEL in self.model_name:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_2_SOURCE, DEMUCS_2_SOURCE_MAPPER, 2
        else:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4

        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = secondary_stem(self.primary_stem)
            
    def get_model_data(self, model_hash_dir, hash_mapper:dict):
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

            return self.get_model_data_from_popup()

    def change_model_data(self):
        if self.is_get_hash_dir_only:
            return None
        else:
            return self.get_model_data_from_popup()

    def get_model_data_from_popup(self):
        if self.is_dry_check:
            return None
            
        if not self.is_change_def:
            # confirm = messagebox.askyesno(
            #     title=UNRECOGNIZED_MODEL[0],
            #     message=f'"{self.model_name}"{UNRECOGNIZED_MODEL[1]}',
            #     parent=root
            # )
            confirm = True
            print(f'"{self.model_name}"{UNRECOGNIZED_MODEL[1]}')
            print("Force confirm: ", confirm)
            if not confirm:
                return None
        
        if self.process_method == VR_ARCH_TYPE:
            root.pop_up_vr_param(self.model_hash)
            return root.vr_model_params
        elif self.process_method == MDX_ARCH_TYPE:
            root.pop_up_mdx_model(self.model_hash, self.model_path)
            return root.mdx_model_params

    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)
                
        #print(self.model_name," - ", self.model_hash)

class Ensembler():
    def __init__(self, is_manual_ensemble=False):
        self.is_save_all_outputs_ensemble = root.is_save_all_outputs_ensemble_var
        chosen_ensemble_name = '{}'.format(root.chosen_ensemble_var.replace(" ", "_")) if not root.chosen_ensemble_var == CHOOSE_ENSEMBLE_OPTION else 'Ensembled'
        ensemble_algorithm = root.ensemble_type_var.partition("/")
        ensemble_main_stem_pair = root.ensemble_main_stem_var.partition("/")
        time_stamp = round(time.time())
        self.audio_tool = MANUAL_ENSEMBLE
        self.main_export_path = root.export_path_var
        self.main_input_path = root.main_input_path
        self.chosen_ensemble = f"_{chosen_ensemble_name}" if root.is_append_ensemble_name_var else ''
        ensemble_folder_name = self.main_export_path if self.is_save_all_outputs_ensemble else ENSEMBLE_TEMP_PATH
        self.ensemble_folder_name = os.path.join(ensemble_folder_name, '{}_Outputs_{}'.format(chosen_ensemble_name, time_stamp))
        self.is_testing_audio = f"{time_stamp}_" if root.is_testing_audio_var else ''
        self.primary_algorithm = ensemble_algorithm[0]
        self.secondary_algorithm = ensemble_algorithm[2]
        self.ensemble_primary_stem = ensemble_main_stem_pair[0]
        self.ensemble_secondary_stem = ensemble_main_stem_pair[2]
        self.is_normalization = root.is_normalization_var
        self.is_wav_ensemble = root.is_wav_ensemble_var
        self.wav_type_set = root.wav_type_set
        self.mp3_bit_set = root.mp3_bit_set_var
        self.save_format = root.save_format_var
        if not is_manual_ensemble and not USE_IN_MEMORY_FS_TO_CACHE_INTERMEDIATE_RESULTS:
            os.mkdir(self.ensemble_folder_name)

    def ensemble_outputs(self, audio_file_base, export_path, stem, is_4_stem=False, is_inst_mix=False, intermediate_folders=''):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        if is_4_stem:
            algorithm = root.ensemble_type_var
            stem_tag = stem
        else:
            if is_inst_mix:
                algorithm = self.secondary_algorithm
                stem_tag = f"{self.ensemble_secondary_stem} {INST_STEM}"
            else:
                algorithm = self.primary_algorithm if stem == PRIMARY_STEM else self.secondary_algorithm
                stem_tag = self.ensemble_primary_stem if stem == PRIMARY_STEM else self.ensemble_secondary_stem

        stem_outputs = self.get_files_to_ensemble(folder=export_path, prefix=audio_file_base, suffix=f"_({stem_tag}).wav")
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}.{stem_tag}"
        if not USE_IN_MEMORY_FS_TO_CACHE_INTERMEDIATE_RESULTS:
            stem_save_path = os.path.join('{}'.format(self.main_export_path), intermediate_folders, '{}.wav'.format(audio_file_output))
        else:
            stem_save_path = os.path.join('{}'.format(self.main_export_path), intermediate_folders, f'{audio_file_output}.{self.save_format.lower()}')
        
        #print("get_files_to_ensemble: ", stem_outputs)
        
        if len(stem_outputs) > 1:
            spec_utils.ensemble_inputs(stem_outputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path, is_wave=self.is_wav_ensemble)
            save_format(stem_save_path, self.save_format, self.mp3_bit_set) # not use if in memory fs
        
        if USE_IN_MEMORY_FS_TO_CACHE_INTERMEDIATE_RESULTS: return
        
        if self.is_save_all_outputs_ensemble:
            for i in stem_outputs:
                save_format(i, self.save_format, self.mp3_bit_set)
        else:
            for i in stem_outputs:
                try:
                    os.remove(i)
                except Exception as e:
                    print(e)

    def ensemble_manual(self, audio_inputs, audio_file_base, is_bulk=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        is_mv_sep = True
        
        if is_bulk:
            number_list = list(set([os.path.basename(i).split("_")[0] for i in audio_inputs]))
            for n in number_list:
                current_list = [i for i in audio_inputs if os.path.basename(i).startswith(n)]
                audio_file_base = os.path.basename(current_list[0]).split('.wav')[0]
                stem_testing = "instrum" if "Instrumental" in audio_file_base else "vocals"
                if is_mv_sep:
                    audio_file_base = audio_file_base.split("_")
                    audio_file_base = f"{audio_file_base[1]}_{audio_file_base[2]}_{stem_testing}"
                self.ensemble_manual_process(current_list, audio_file_base, is_bulk)
        else:
            self.ensemble_manual_process(audio_inputs, audio_file_base, is_bulk)
            
    def ensemble_manual_process(self, audio_inputs, audio_file_base, is_bulk):
        
        algorithm = root.choose_algorithm_var
        algorithm_text = "" if is_bulk else f"_({root.choose_algorithm_var})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}{}{}.wav'.format(self.is_testing_audio, audio_file_base, algorithm_text))
        spec_utils.ensemble_inputs(audio_inputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path, is_wave=self.is_wav_ensemble)
        save_format(stem_save_path, self.save_format, self.mp3_bit_set)

    def get_files_to_ensemble(self, folder="", prefix="", suffix=""):
        """Grab all the files to be ensembled"""
        if USE_IN_MEMORY_FS_TO_CACHE_INTERMEDIATE_RESULTS:
            filelist = list(in_memory_fs.keys())
            if not DISABLE_LOGGING: print("In-Memory FS: ", filelist)
            if not folder.endswith('/'): folder += '/'
            filelist_return = []
            for file in filelist:
                if file.replace(folder, '').startswith(prefix) and file.endswith(suffix):
                    filelist_return.append(file)
            return filelist_return

        return [os.path.join(folder, i) for i in os.listdir(folder) if i.startswith(prefix) and i.endswith(suffix)]

    def combine_audio(self, audio_inputs, audio_file_base):
        save_format_ = lambda save_path:save_format(save_path, root.save_format_var, root.mp3_bit_set_var)
        spec_utils.combine_audio(audio_inputs, 
                                 os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}"), 
                                 self.wav_type_set,
                                 save_format=save_format_)


def find_audios(parent_dir, exts=['.wav', '.mp3', '.flac', '.webm', '.mp4']):
    audio_files = []
    for root, dirs, files in os.walk(parent_dir, followlinks=True):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                audio_files.append(os.path.join(root, file))
                if len(audio_files) % 10000 == 0:
                    print(f"found {len(audio_files)} audio files...")
    return audio_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', "-i", type=str, 
                        required=True, 
                        help='Input path, can be a single audio file, or a .txt file containing a list of audio files, or a directory containing audio files')
    parser.add_argument('--output', "-o", type=str, 
                        required=True, 
                        help='Output dir, a directory where the separated stems will be saved')
    parser.add_argument("--resume", '-r', action="store_true")
    parser.add_argument("--total_shard", '-t', type=int, default=1)
    parser.add_argument("--cur_shard", '-s', type=int, default=0)
    parser.add_argument("--cuda_idx", '-g', type=int, default=0)
    parser.add_argument("--save_format", type=str, default='MP3', choices=[WAV, FLAC, MP3])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1], help="0: MIN_MIX, 1: MAX_MIN. 0 gives cleaner vocals.")
    parser.add_argument("--read_from_tos", action="store_true")
    parser.add_argument("--prefix_to_replace", type=str, default=None, help="this string in the file path will be replaced by output dir, used during keep folder structure.")

    args = parser.parse_args()


    os.makedirs(args.output, exist_ok=True)

    # get filelist
    if os.path.isdir(args.input):
        fl_saved = os.path.join(args.output, "filelist.txt")
        if os.path.exists(fl_saved):
            with open(fl_saved, "r") as fp:
                audio_files = fp.read().splitlines()
            print(f"file list loaded, len {len(audio_files)}")
        else:
            print("file list not found, searching audio files...")
            audio_files = find_audios(args.input)
            audio_files.sort()
            with open(fl_saved, "w") as fp:
                fp.write("\n".join(audio_files))
            print(f"file list saved, len {len(audio_files)}")

    elif os.path.isfile(args.input):
        if args.input.endswith('.txt'):
            with open(args.input, 'r') as f:
                audio_files = f.read().splitlines()
        else:
            audio_files = [args.input]
    else:
        raise ValueError("Invalid input path")
    
    # shuffle
    if args.shuffle:
        if args.shuffle_seed is not None:
            random.seed(args.shuffle_seed)
        random.shuffle(audio_files)
    
    print(f"current shard: {args.cur_shard + 1}/{args.total_shard}")
    cur_audio_files = audio_files[args.cur_shard * len(audio_files) // args.total_shard : (args.cur_shard + 1) * len(audio_files) // args.total_shard]
    # fix final shard
    if args.cur_shard == args.total_shard - 1:
        cur_audio_files = audio_files[args.cur_shard * len(audio_files) // args.total_shard :]

    if args.read_from_tos:
        from tos_audio_read import *

    print(args)

    # PROFILE_FLG=True
    # import cProfile

    # def main():
    #     cur_audio_files = audio_files[:1]

    #     # fake mainwindow
    #     global root
    #     root = MainWindow(
    #         inputPaths=cur_audio_files, 
    #         export_path=args.output,
    #         cuda_idx=args.cuda_idx,
    #         save_format=args.save_format,
    #         resume=args.resume,
    #         ensemble_type_var=args.mode,
    #         main_input_path=args.input if args.prefix_to_replace is None else args.prefix_to_replace,
    #         read_from_tos=args.read_from_tos
    #         )
        
    #     root.process_start()

    # if PROFILE_FLG:
    #     cProfile.run('main()', sort="cumulative")

    root = MainWindow(
            inputPaths=cur_audio_files, 
            export_path=args.output,
            cuda_idx=args.cuda_idx,
            save_format=args.save_format,
            resume=args.resume,
            ensemble_type_var=args.mode,
            main_input_path=args.input if args.prefix_to_replace is None else args.prefix_to_replace,
            read_from_tos=args.read_from_tos
            )
        
    root.process_start()
