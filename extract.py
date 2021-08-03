import argparse
from pathlib import Path

import essentia.standard as ess
import numpy as np

SAMPLE_RATE = 16000


def extract(audio_file: Path, model_file: Path, output_file: Path, algorithm: str, output_layer: str,
            hop_size: int, accumulate: bool, info: bool):
    try:
        algorithm_class = getattr(ess, algorithm)
    except AttributeError:
        print(f'No algorithm {algorithm} in essentia.standard')
        return

    audio = ess.MonoLoader(filename=str(audio_file), sampleRate=SAMPLE_RATE)()
    embeddings = algorithm_class(graphFilename=str(model_file), patchHopSize=hop_size, output=output_layer,
                                 accumulate=accumulate)(audio)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)

    if info:
        print(embeddings.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_file', type=Path, help='input audio file (e.g. mp3)')
    parser.add_argument('model_file', type=Path, help='model .pb file')
    parser.add_argument('output_file', type=Path, help='output .npy file')
    parser.add_argument('--algorithm', type=str, default='TensorflowPredictMusiCNN', help='essentia algorithm')
    parser.add_argument('--output-layer', type=str, default='model/Sigmoid', help='name of the layer')
    parser.add_argument('--hop-size', type=int, default=0, help='hop size')
    parser.add_argument('--accumulate', action='store_true', help='use same tf session for the audio file')
    parser.add_argument('--info', action='store_true', help='show info about extracted layer')
    args = parser.parse_args()

    extract(**vars(args))
