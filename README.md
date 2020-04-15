# End-to-End Speech Recognition System Using Connectionist Temporal Classification
Automatic speech recognition (ASR) system implementation that utilizes the 
[connectionist temporal classification (CTC)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.75.6306)
cost function.
It's inspired by Baidu's
[Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567)
and
[Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
papers.
The system is trained on a combined corpus, containing 900+ hours.
It achieves a word error rate (WER) of 12.6% on the test dataset, without the use of an external
language model.


## Contents
<!-- TOC_START -->

* [Contents](#contents)
* [Installation](#installation)
  * [Arch Linux](#arch-linux)
  * [Ubuntu](#ubuntu)
* [Configuration](#configuration)
* [Corpus](#corpus)
  * [CSV](#csv)
  * [Free Speech Corpora](#free-speech-corpora)
  * [Corpus Statistics](#corpus-statistics)
* [Usage](#usage)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Prediction](#prediction)

<!-- TOC_END -->

![Deep Speech 1 and 2 network architectures](images/network-architectures.png)

(a) shows the Deep Speech (1) model and (b) a version of the Deep Speech 2 model architecture. 


## Installation
The system was tested on Arch Linux and Ubuntu 16.04, with Python version 3.5+ and the 1.12.0 
version of [TensorFlow](https://www.tensorflow.org/). It's highly recommended to use TensorFlow 
with GPU support for training.


### Arch Linux
```terminal
# Install dependencies.
sudo pacman -S sox python-tensorflow-opt-cuda tensorbaord

# Install optional dependencies. LaTeX is only required to plot nice looking graphs.
sudo pacman -S texlive-most

# Clone reposetory and install Python depdendencies.
git clone https://github.com/mdangschat/ctc-asr.git
cd speech
git checkout <release_tag>

# Setup optional virtual environment.
pip install -r requirements.txt
```


### Ubuntu
Be aware that the [`requirements.txt`](requirements.txt) file lists `tensorflow` as dependency, 
if you install TensorFlow through [pip](https://pypi.org/project/pip/) consider removing it as 
dependency and install `tensorflow-gpu` instead.
It could also be worth it to [build TensorFlow from source](https://www.tensorflow.org/install/source).

```terminal
# Install dependencies.
sudo apt install python3-tk sox libsox-fmt-all

# Install optional dependencies. LaTeX is only required to plot nice looking graphs.
sudo apt install texlive

# Clone reposetory and install Python depdendencies. Don't forget to use tensorflow-gpu.
git clone https://github.com/mdangschat/ctc-asr.git
cd speech
git checkout <release_tag>

# Setup optional virtual environment.
pip3 install -r requirements.txt
```


## Configuration
The network architecture and training parameters can be configured by adding the appropriate flags
or by directly editing the [`asr/params.py`](asr/params.py) configuration file.
The default configuration requires quite a lot of VRAM (about 16 GB), consider reducing the number of units per
layer (`num_units_dense`, `num_units_rnn`) and the amount of RNN layers (`num_layers_rnn`).


## Corpus
There is list of some [free speech corpora](#free-speech-corpora) at the end of this section.
However, the corpus is not part of this repository and has to be acquired by each user.
For a quick start there is the [speech-corpus-dl](https://github.com/mdangschat/speech-corpus-dl) 
helper, that downloads a few free corpora, prepares the data and creates a merged corpus.

All audio files have to be 16 kHz, mono, WAV files.
For my trainings, I removed examples shorter than 0.7 and longer than 17.0 seconds.
Additionally, TEDLIUM examples with labels of fewer than 5 words have also been removed.

The following tree shows a possible structure for the required directories:
```terminal
./ctc-asr
├── asr
    ├── [...]
├── LICENSE
├── README.md
├── requirements.txt
├── testruns.md
./ctc-asr-checkpoints
└── 3c2r2d-rnn
    ├── [...]
./speech-corpus
├── cache
├── corpus
│   ├── cvv2
│   ├── LibriSpeech
│   ├── tatoeba_audio_eng
│   └── TEDLIUM_release2
├── corpus.json
├── dev.csv
├── test.csv
└── train.csv
```
Assuming that this repository is cloned into `some/folder/ctc-asr`, then by default
the CSV files are expected to be in `some/folder/speech-corpus` and the audio files in
`some/folder/speech-corpus/corpus`.
TensorFlow checkpoints are written into `some/folder/ctc-asr-checkpoints`.
Both folders (`ctc-asr-checkpoints` and `speech-corpus`) must exist, they can be changed
in the [asr/params.py](asr/params.py) file.


### CSV
The CSV files (e.g. train.csv) have the following format:
```csv
path;label;length
relative/path/to/example;lower case transcription without puntuation;3.14159265359
[...]
```
Where `path` is the relative WAV path from the `DATA_DIR/corpus/` directory (String).
By default, `label` is the lower case transcription without punctuation (String).
Finally, `length` is the audio length in seconds (Float).


### Free Speech Corpora
* [Common Voice](https://voice.mozilla.org/en/new) (v1)
* [LibriSpeech ASR Corpus](http://www.openslr.org/12/)
* [Tatoeba](https://tatoeba.org/eng/)
* [TED-Lium](http://www.openslr.org/19/) (v2)
* [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)


### Corpus Statistics
```terminal
ipython python/dataset/word_counts.py 
Calculating statistics for /home/gpuinstall/workspace/ctc-asr/data/train.csv
Word based statistics:
        total_words = 10,069,671
        number_unique_words = 81,161
        mean_sentence_length = 14.52 words
        min_sentence_length = 1 words
        max_sentence_length = 84 words
        Most common words:  [('the', 551055), ('to', 306197), ('and', 272729), ('of', 243032), ('a', 223722), ('i', 192151), ('in', 149797), ('that', 146820), ('you', 144244), ('it', 118133)]
        27416 words occurred only 1 time; 37,422 words occurred only 2 times; 49,939 words occurred only 5 times; 58,248 words occurred only 10 times.

Character based statistics:
        total_characters = 52,004,043
        mean_label_length = 75.00 characters
        min_label_length = 2 characters
        max_label_length = 422 characters
        Most common characters: [(' ', 9376326), ('e', 5264177), ('t', 4205041), ('o', 3451023), ('a', 3358945), ('i', 2944773), ('n', 2858788), ('s', 2624239), ('h', 2598897), ('r', 2316473), ('d', 1791668), ('l', 1686896), ('u', 1234080), ('m', 1176076), ('w', 1052166), ('c', 999590), ('y', 974918), ('g', 888446), ('f', 851710), ('p', 710252), ('b', 646150), ('v', 421126), ('k', 387714), ('x', 62547), ('j', 61048), ('q', 34558), ('z', 26416)]
        Most common characters: [' ', 'e', 't', 'o', 'a', 'i', 'n', 's', 'h', 'r', 'd', 'l', 'u', 'm', 'w', 'c', 'y', 'g', 'f', 'p', 'b', 'v', 'k', 'x', 'j', 'q', 'z']
```


## Usage
### Training
Start training by invoking `asr/train.py`.
Use `asr/train.py -- --delete` to start a clean run and remove the old checkpoints.
Please note that all commands are expected to be executed from the projects root folder.
The additional `--` before the actual flags begin is used to indicate the end of IPython flags.

The training progress can be monitored using Tensorboard.
To start Tensorboard use `tensorboard --logdir <checkpoint_directory>`.
By default it can then be accessed via [localhost:6006](http://localhost:6006).


### Evaluation
Evaluate the current model by invoking `asr/evaluate.py`.
Use `asr/evaluate.py -- --dev` to run on the development dataset, instead of the test set.


### Prediction
To evaluate a given 16 kHz, mono WAV file use `asr/predict.py --input <wav_path>`.

