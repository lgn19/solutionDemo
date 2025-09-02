import os
import sys
import argparse
import json
from typing import List
from pathlib import Path
import tqdm
import random
import multiprocessing.dummy as mp
import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf
"""
sound/p225
genertated
--n_voices
1
--n_outputs
300
--mic_radius
0.32
--n_mics
16
"""

# Mean and STD of the signal peak
FG_VOL_MIN = 0.15
FG_VOL_MAX = 0.4

BG_VOL_MIN = 0.2
BG_VOL_MAX = 0.5


def handle_error(e):
    print(e)


def get_voices(args):
    # Make sure we dont get an empty sequence
    success = False
    while not success:
        voice_files = random.sample(args.all_voices, args.n_voices)
        # Save the identity also. This is VCTK specific
        success = True
        voices_data = []
        for voice_file in voice_files:
            voice_identity = str(voice_file).split("/")[-1].split("_")[0]
            voice, _ = librosa.core.load(voice_file, sr=args.sr, mono=True)
            voice, _ = librosa.effects.trim(voice, top_db=60)
            # print(voice.shape,voice_file)
            if voice.std() == 0:
                success = False
            voices_data.append((voice, voice_identity))
    return voices_data


def generate_mic_array(room, mic_positions):
    """
    Generate a list of Microphone objects with fixed 3D positions.
    """
    R = np.array(mic_positions).T  # Transpose for pra.MicrophoneArray
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))


def generate_sample(args: argparse.Namespace, bg: np.ndarray, idx: int) -> int:
    # [1] Load voice and prepare output path
    output_prefix_dir = os.path.join(args.output_path, '{:05d}'.format(idx))
    Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)
    voices_data = get_voices(args)

    # [2] Background sound preparation
    total_samples = int(args.duration * args.sr)
    if bg is not None:
        bg_length = len(bg)
        bg_start_idx = np.random.randint(bg_length - total_samples)
        sample_bg = bg[bg_start_idx:bg_start_idx + total_samples]

    # 继续添加声源及其位置
    voice_positions = []
    all_fg_signals = []
    for voice_idx in range(args.n_voices):

        corner = np.array([[0, 0], [8, 0], [8, 5], [0, 5]]).T
        room = pra.Room.from_corners(corner,
                                     fs=args.sr,  # 采样率
                                     max_order=6,  # 最大反射次数
                                     absorption=0.1)
        room.extrude(3.)
        room.air_absorption = True

        mic_positions = [[1.8, 0.6, 1], [1.8667, 0.6, 1], [1.9333, 0.6, 1], [2, 0.6, 1],
                         [2.4, 0.6, 1], [2.4667, 0.6, 1], [2.5333, 0.6, 1], [2.6, 0.6, 1],
                         [1.8, 0.6, 2], [1.8667, 0.6, 2], [1.9333, 0.6, 2], [2, 0.6, 2],
                         [2.4, 0.6, 2], [2.4667, 0.6, 2], [2.5333, 0.6, 2], [2.6, 0.6, 2]]


        mic_araary = generate_mic_array(room, mic_positions)

        voice_loc = [
            np.random.uniform(low=1.4, high=4),
            np.random.uniform(low=1, high=2),
            np.random.uniform(low=0.8, high=2.2)
        ]
        voice_positions.append(voice_loc)
        room.add_source(voice_loc, signal=voices_data[voice_idx][0])
        room.image_source_model(use_libroom=True)
        room.simulate()
        fg_signals = room.mic_array.signals[:, :total_samples]
        fg_target = np.random.uniform(FG_VOL_MIN, FG_VOL_MAX)
        fg_signals = fg_signals * fg_target / abs(fg_signals).max()
        all_fg_signals.append(fg_signals)

    # 背景音处理和保存，保持与之前相同
    # BG
    if bg is not None:
        bg_radius = np.random.uniform(low=10.0, high=20.0)
        bg_theta = np.random.uniform(low=0, high=2 * np.pi)
        bg_loc = [bg_radius * np.cos(bg_theta), bg_radius * np.sin(bg_theta)]

        # Bg should be further away to be diffuse
        left_wall = np.random.uniform(low=-40, high=-20)
        right_wall = np.random.uniform(low=20, high=40)
        top_wall = np.random.uniform(low=20, high=40)
        bottom_wall = np.random.uniform(low=-40, high=-20)
        corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                            [right_wall, top_wall], [right_wall, bottom_wall]]).T
        absorption = np.random.uniform(low=0.5, high=0.99)
        room = pra.Room.from_corners(corners,
                                     fs=args.sr,
                                     max_order=10,
                                     absorption=absorption)
        mic_array = generate_mic_array(room, args.mic_radius, args.n_mics)
        room.add_source(bg_loc, signal=sample_bg)

        room.image_source_model(use_libroom=True)
        room.simulate()
        bg_signals = room.mic_array.signals[:, :total_samples]
        bg_target = np.random.uniform(BG_VOL_MIN, BG_VOL_MAX)
        bg_signals = bg_signals * bg_target / abs(bg_signals).max()

    # Save
    for mic_idx in range(args.n_mics):
        output_prefix = str(
            Path(output_prefix_dir) / "mic{:02d}_".format(mic_idx))

        # Save FG
        all_fg_buffer = np.zeros((total_samples))
        for voice_idx in range(args.n_voices):
            curr_fg_buffer = np.pad(all_fg_signals[voice_idx][mic_idx],
                                    (0, total_samples))[:total_samples]
            sf.write(output_prefix + "voice{:02d}.wav".format(voice_idx),
                     curr_fg_buffer, args.sr)
            all_fg_buffer += curr_fg_buffer

        if bg is not None:
            bg_buffer = np.pad(bg_signals[mic_idx],
                               (0, total_samples))[:total_samples]
            sf.write(output_prefix + "bg.wav", bg_buffer, args.sr)

            sf.write(output_prefix + "mixed.wav", all_fg_buffer + bg_buffer,
                     args.sr)
        else:
            sf.write(output_prefix + "mixed.wav", all_fg_buffer,
                     args.sr)

    # [6]
    metadata = {}
    for voice_idx in range(args.n_voices):
        metadata['voice{:02d}'.format(voice_idx)] = {
            'position': voice_positions[voice_idx],
            'speaker_id': voices_data[voice_idx][1]
        }

    if bg is not None:
        metadata['bg'] = {'position': bg_loc}

    metadata_file = str(Path(output_prefix_dir) / "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    # Preload background to save time
    if args.input_background_path:
        background, _ = librosa.core.load(args.input_background_path,
                                          sr=args.sr,
                                          mono=True)
    else:
        background = None

    all_voices = Path(args.input_voice_dir).rglob('*.wav')
    args.all_voices = list(all_voices)
    if len(args.all_voices) == 0:
        raise ValueError("No voice files found")

    pbar = tqdm.tqdm(total=args.n_outputs)
    pool = mp.Pool(args.n_workers)
    callback_fn = lambda _: pbar.update()
    for i in range(args.n_outputs):
        pool.apply_async(generate_sample,
                         args=(args, background, i),
                         callback=callback_fn,
                         error_callback=handle_error)
    pool.close()
    pool.join()
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_voice_dir',
                        type=str,
                        default="/sound/p225"
                        )
    parser.add_argument('output_path', type=str,
                        default="genertated")
    parser.add_argument('--input_background_path',
                        type=str)
    parser.add_argument('--n_mics', type=int, default=16)
    parser.add_argument('--mic_radius',
                        default=.03231,
                        type=float,
                        help="Radius of the mic array in meters")
    parser.add_argument('--n_voices', type=int, default=1)
    parser.add_argument('--n_outputs', type=int, default=300)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--duration', type=float, default=3.0)
    main(parser.parse_args())
