import collections
import sys
import librosa
import numpy
import random


MU = 1800

def read_wave_to_frames(path, sr=16000, frame_duration=10):
    wav, orig_sr = librosa.load(path, sr=sr, mono=True, res_type='polyphase')
    wav = (wav * 2**15).astype(numpy.int16)
    wav_bytes = wav.tobytes()
    frames = frame_generator(frame_duration, wav_bytes, sr)
    return frames, wav



class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_generator(frames, sr, vad):
    vad_info = []
    for frame in frames:
        vad_info.append(vad.is_speech(frame.bytes, sr))
    return vad_info


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


class ActivateInfo:
    def __init__(self, active, duration, start_pos, end_pos, keep=True):
        self.active = active
        self.duration = duration
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.keep = keep

    def __add__(self, x):
        return x + self.duration
    
    def __repr__(self) -> str:
        return f"{self.active} {self.start_pos}, {self.end_pos}"


class SegmentInfo:
    def __init__(self, type="raw", duration=0, start_pos=0, end_pos=0, frame_duration=10):
        self.type = type
        self.duration = duration
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.frame_duration = frame_duration

    def get_wav_seg(self, wav: numpy.array, sr: int, frame_duration: int=None):
        fd = frame_duration if frame_duration is not None else self.frame_duration
        sample_pre_frame = fd*sr/1000
        if self.type == "pad":
            return numpy.zeros((int(sample_pre_frame*self.duration), ), dtype=numpy.int16)
        return wav[int(self.start_pos*sample_pre_frame):int((self.end_pos*sample_pre_frame))]

    def __repr__(self) -> str:
        if self.type == "raw":
            text = f"{self.start_pos*self.frame_duration}:{self.end_pos*self.frame_duration}"
        else:
            text = f"[{self.duration*self.frame_duration}]"
        return text


def get_sil_segments(active_info: ActivateInfo, sil_frame: int, attach_pos: str="mid") -> list:
    if active_info.duration >= sil_frame:
        if attach_pos == "tail":
            seg = [SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.start_pos+sil_frame)]
        elif attach_pos == "head":
            seg = [SegmentInfo(start_pos=active_info.end_pos-sil_frame, end_pos=active_info.end_pos)]
        elif attach_pos == "mid":
            seg = [
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.start_pos+sil_frame // 2-1),
                SegmentInfo(start_pos=active_info.end_pos-sil_frame // 2+1, end_pos=active_info.end_pos),
            ]
        else:
            raise NotImplementedError
    else:
        if attach_pos == "tail":
            seg = [
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.end_pos),
                SegmentInfo(type="pad", duration=sil_frame-active_info.duration),
            ]
        elif attach_pos == "head":
            seg = [
                SegmentInfo(type="pad", duration=sil_frame-active_info.duration),
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.end_pos),
            ]
        elif attach_pos == "mid":
            seg = [
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.end_pos),
            ]
        else:
            raise NotImplementedError
    return seg


def merge_segment(segment: list) -> list:
    new_segment = []
    last_s = None
    for s in segment:
        s: SegmentInfo
        if s.type == "pad":
            if last_s is not None:
                new_segment.append(last_s)
                last_s = None
            new_segment.append(s)
            continue
        if last_s is None:
            last_s = s
        else:
            if last_s.end_pos+1 == s.start_pos:
                last_s.end_pos = s.end_pos
            else:
                new_segment.append(last_s)
                last_s = s
    if last_s is not None:
        new_segment.append(last_s)
    return new_segment


def random_frame(min_frame, max_frame):
    #return random.randint(min_frame, max_frame)
    #mu = (max_frame + max_frame + min_frame) / 3
    mu = MU
    sigma = (mu - min_frame) / 3
    length = random.gauss(mu, sigma)
    length = int(min(max(length, min_frame), max_frame))
    return length


def cut_points_generator(
        vad_info, 
        min_active_frame=20, 
        sil_frame=50, 
        sil_mid_frame=100, 
        cut_min_frame=8 * 100, 
        cut_max_frame=20 * 100, 
        is_random_min_frame=False,
    ):
    curr_min_frame = cut_min_frame
    last_active_frame = 0
    is_last_active = False
    for i, is_curr_active in enumerate(vad_info):
        if is_curr_active and not is_last_active:
            last_active_frame = i
        elif not is_curr_active and is_last_active and i - last_active_frame <= min_active_frame:
            for j in range(last_active_frame, i):
                vad_info[j] = False
        is_last_active = is_curr_active

    start_pos = 0
    end_pos = 0
    duration = 0
    is_active = vad_info[0]
    activate_info = []
    for pos, vi in enumerate(vad_info):
        if is_active == vi:
            duration += 1
        else:
            activate_info.append(ActivateInfo(is_active, duration, start_pos, pos-1))
            is_active = vi
            start_pos = pos
            duration = 1
    activate_info.append(ActivateInfo(is_active, duration, start_pos, end_pos))
    segment_info = []
    curr_segment = []
    curr_segment_duration = 0
    max_active_block = len(activate_info)
    for i in range(max_active_block):
        curr_ai = activate_info[i]
        if curr_ai.active:
            if curr_segment_duration == 0:
                if i == 0:
                    curr_segment.append(SegmentInfo("pad", sil_frame))
                else:
                    sil_seg = activate_info[i-1]
                    raw_sil_duration = min(sil_frame, sil_seg.duration // 2)
                    end_pos = sil_seg.end_pos
                    curr_segment = get_sil_segments(
                        ActivateInfo(
                            True, 
                            duration=raw_sil_duration,
                            start_pos=sil_seg.end_pos-raw_sil_duration, 
                            end_pos=sil_seg.end_pos
                        ),
                        sil_frame=sil_frame,
                        attach_pos="head"
                    )
                curr_segment_duration += sil_frame
            next_duration = curr_segment_duration + curr_ai.duration
            curr_ai_seg = SegmentInfo(start_pos=curr_ai.start_pos, end_pos=curr_ai.end_pos)
            if next_duration > cut_max_frame:
                if curr_ai.duration > curr_segment_duration:
                    new_segment = get_sil_segments(activate_info[i-1], sil_frame, "head")
                    new_segment.append(curr_ai_seg)
                    if i < max_active_block - 1:
                        new_segment.extend(get_sil_segments(activate_info[i+1], sil_frame, "tail"))
                    else:
                        new_segment.append(SegmentInfo(type="pad", duration=sil_frame))
                    segment_info.append(merge_segment(new_segment))
                    if is_random_min_frame:
                        curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                    curr_segment = []
                    curr_segment_duration = 0
                else:
                    if curr_segment_duration > 10 * 100:
                        segment_info.append(merge_segment(curr_segment))
                        if is_random_min_frame:
                            curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                    curr_segment = get_sil_segments(activate_info[i-1], sil_frame, "head")
                    curr_segment.append(curr_ai_seg)
                    curr_segment_duration = sil_frame + curr_ai.duration
            elif next_duration > curr_min_frame:
                curr_segment.append(curr_ai_seg)
                if i < max_active_block - 1:
                    curr_segment.extend(get_sil_segments(activate_info[i+1], sil_frame, "tail"))
                else:
                    curr_segment.append(SegmentInfo(type="pad", duration=sil_frame))
                segment_info.append(merge_segment(curr_segment))
                if is_random_min_frame:
                    curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                curr_segment = []
                curr_segment_duration = 0
            else:
                curr_segment.append(curr_ai_seg)
                curr_segment_duration += curr_ai.duration
        else:
            if curr_segment_duration == 0:
                raw_sil_duration = min(sil_frame, curr_ai.duration // 2)
                end_pos = curr_ai.end_pos
                curr_segment = get_sil_segments(
                    ActivateInfo(
                        True, 
                        duration=raw_sil_duration,
                        start_pos=curr_ai.end_pos-raw_sil_duration, 
                        end_pos=curr_ai.end_pos
                    ),
                    sil_frame=sil_frame,
                    attach_pos="head"
                )
                curr_segment_duration += sil_frame 
            else:
                if curr_ai.duration > sil_mid_frame:
                    curr_segment.extend(get_sil_segments(curr_ai, sil_frame, "tail"))
                    segment_info.append(merge_segment(curr_segment))
                    if is_random_min_frame:
                        curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                    curr_segment = []
                    curr_segment_duration = 0
                else:
                    curr_segment.extend(get_sil_segments(curr_ai, sil_mid_frame+1, attach_pos="mid"))
                    curr_segment_duration += min(sil_mid_frame, curr_ai.duration)
    if len(curr_segment) > 3 and curr_segment_duration > 7 * 100:
        if activate_info[-1].active:
            curr_segment.append(SegmentInfo(type="pad", duration=sil_frame))
        segment_info.append(merge_segment(curr_segment))
    return segment_info


def ms_to_mm_ss_xx(milliseconds):
    minutes = milliseconds // 60000
    seconds = (milliseconds % 60000) // 1000
    milliseconds_part = (milliseconds % 1000) // 10
    return f"[{minutes:02d}:{seconds:02d}.{milliseconds_part:02d}]"

def process_timestamp(timestamp, cut_min_ms):
    start, end = map(int, timestamp.split(":"))
    duration = end - start
    
    if duration >= cut_min_ms:
        return ms_to_mm_ss_xx(start)
    else:
        return None


def cut_points_storage_generator(raw_vad_info, cut_points: list, cut_min_ms=600) -> tuple:
    raw_vad_content = " ".join(["1" if i else "0" for i in raw_vad_info])
    content = []
    for cut_point in cut_points:
        time_ranges = []
        for s in cut_point:
            s_str = str(s)
            if not s_str.startswith('['):
                result = process_timestamp(s_str, cut_min_ms)
                if result is not None:
                    time_ranges.append(result)
        if time_ranges:
            content.append(time_ranges[0]) 
    return raw_vad_content, "\n".join(content)

def wavs_generator(raw_wav: numpy.array, cut_points: list, filename: str, sr: int, frame_duration: int) -> list:
    wavs = []
    for idx, cp in enumerate(cut_points):
        clip = numpy.concatenate(
            [s.get_wav_seg(raw_wav, sr, frame_duration) for s in cp],
            axis=0
        )
        wavs.append((clip, f"{filename}_{idx}_{int(clip.shape[0]/sr*1000)}.wav"))
    return wavs
    