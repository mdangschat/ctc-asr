# End-to-End Speech Recognition System Using Connectionist Temporal Classification
Automatic speech recognition (ASR) system implementation that utilizes the 
[connectionist temporalclassification (CTC)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.75.6306)
cost function.
It is inspired by Baidu's
[Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567)
and
[Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)
papers.
The system is trained on a combined corpus, containing 900+ hours.
It achieves a word error rate (WER) of 12.6% on the test dataset, without the use of an external
language model.

![Deep Speech 1 and 2 network architectures](images/network-architectures.png)

(a) shows the Deep Speech (1) model and (b) a version of the Deep Speech 2 model architecture. 


## Contents
* [Installation](#installation)
  * [Arch Linux](#arch-linux)
  * [Ubuntu](#ubuntu)
  * [WarpCTC](#compile-and-install-with-warpctc-support)
* [Configuration](#configuration)
* [Datasets](#datasets)
* [Training](#training)
* [Evaluation](#evaluation)
* [License](#license)


## Installation
The system was tested on Arch Linux and Ubuntu 16.04, with Python version 3.5+ and the 1.9.0 version
of [TensorFlow](https://www.tensorflow.org/). It is highly recommended to use TensorFlow with GPU
support for training.


### Arch Linux
```sh
# Install dependencies.
sudo pacman -S sox python-tensorflow-opt-cuda tensorbaord

# Install optional dependencies.
sudo pacman -S texlive-most

# Clone reposetory and install Python depdendencies.
git clone <URL>
cd speech
pip install -r requirements.txt
```

### Ubuntu
Be aware that the [requirements.txt](requirements.txt) lists `tensorflow` as dependency, if you
install TensorFlow through [pip](https://pypi.org/project/pip/) consider removing it and install 
`tensorflow-gpu` manually.
Based on my experience it is worth the effort to 
[build TensorFlow from source](https://www.tensorflow.org/install/source).

```sh
# Install dependencies.
sudo apt install python3-tk sox libsox-fmt-all

# Install optional dependencies.
sudo apt install texlive

# Clone reposetory and install Python depdendencies. Don't forget tensorflow-gpu.
git clone <URL>
cd speech
pip3 install -r requirements.txt
```


### Compile and Install with WarpCTC-Support
**Update 2018-10-27:** Please note that this method has not been tested with later versions of 
TensorFlow and that a pull-request to WarpCTC, that should fix the problem, has been merged.


#### Compile TensorFlow
```sh
# Tensorflow
git clone https://github.com/tensorflow/tensorflow
cd tensorflow

# Checkout the desired version (e.g. rolling `r1.9` or release `v1.9.0`).
git checkout v1.9.0

# Run config wizard
./configure

# Build tensorflow
bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package

# Build pip installer
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install or upgrade package.
sudo pip install /tmp/tensorflow_pkg/tensorflow-X.X.X-cp35-cp35m-linux_x86_64.whl
sudo pip install -U /tmp/tensorflow_pkg/tensorflow-X.X.X-cp35-cp35m-linux_x86_64.whl
```


#### Compile WarpCTC
```sh
# Back to base folder
cd ..

# Set environment variables.
export CUDA_HOME="/usr/local/cuda"
export TENSORFLOW_SRC_PATH=<tensorflow_repo>
export WARP_CTC_PATH="<path_to_repo>/warp-ctc/build"

git clone https://vcs.zwuenf.org/mdangschat/warp-ctc.git
cd warp-ctc

mkdir build && cd build
cmake ../
make

# Install TensorFlow python bindings/
cd ../tensorflow_binding
python setup.py install

# Test Warp CTC.
python setup.py test
```

Reference [installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources) 
and 
[TensorFlow binding for WarpCTC](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding) 
for further information.


## Configuration
The network architecture and training parameters can be configured by adding the appropriate flags
or by directly editing the [params.py](python/params.py) configuration file.


## Datasets
The following datasets were used for training and are listed in the `data` directory, however, the
individual datasets are not part of the repository and have to be acquired by each user.

* [Common Voice](https://voice.mozilla.org/en/new) (v1)
* [LibriSpeech ASR Corpus](http://www.openslr.org/12/)
* [Tatoeba](https://tatoeba.org/eng/)
* [TED-Lium](http://www.openslr.org/19/) (v2)
* [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)

The test dataset consists of all clean training subsets from those datasets.
Only the LibriSpeech clean dev set is used as the validation/development set and the LibriSpeech
and Common Voice clean test sets are used as testing dataset. 
The ASR system works on 16 kHz mono `.wav` files.


### `train.txt` 906+ Hours
Examples shorter than 0.7 and longer than 17.0 seconds have been removed.
Tedlium samples with labels shorter than 5 words have been removed.
`train.txt` is sorted by feature sequence length in ascending order.

* `commonvoice_train.txt`
* `librispeech_train.txt`
* `tatoeba_train.txt`
* `tedlium_train.txt`
* `timit_train.txt`


### `dev.txt`
* `librispeech_dev.txt`


### `test.txt`
* `commonvoice_test.txt`
* `librispeech_test.txt`


### Statistics
```
/usr/bin/python3 -u /home/marc/workspace/speech/asr/dataset_util/word_counts.py
Calculating statistics for /mnt/storage/workspace/speech/data/train.txt
Word based statistics:
	total_words = 8,912,133
    number_unique_words = 81,090
	mean_sentence_length = 15.61 words
	min_sentence_length = 1 words
	max_sentence_length = 84 words
	Most common words:  [('the', 461863), ('to', 270959), ('and', 246560), ('of', 220573), ('a', 198632), ('i', 171289), ('in', 135662), ('that', 130372), ('you', 127414), ('tom', 114623)]
	27437 words occurred only 1 time; 37473 words occurred only 2 times; 50106 words occurred only 5 times; 58618 words occurred only 10 times.

Character based statistics:
	total_characters = 46,122,731
	mean_label_length = 80.80 characters
	min_label_length = 2 characters
	max_label_length = 422 characters
	Most common characters: [(' ', 8341279), ('e', 4631650), ('t', 3712085), ('o', 3082222), ('a', 2973836), ('i', 2629625), ('n', 2542519), ('s', 2332114), ('h', 2266656), ('r', 2067363), ('d', 1575679), ('l', 1504091), ('u', 1098686), ('m', 1047460), ('w', 928518), ('c', 900106), ('y', 856603), ('g', 783085), ('f', 764710), ('p', 633259), ('b', 563912), ('v', 377197), ('k', 342523), ('x', 56864), ('j', 54972), ('q', 32031), ('z', 23686)]
	Most common characters: [' ', 'e', 't', 'o', 'a', 'i', 'n', 's', 'h', 'r', 'd', 'l', 'u', 'm', 'w', 'c', 'y', 'g', 'f', 'p', 'b', 'v', 'k', 'x', 'j', 'q', 'z']
```


## Training
Start training by invoking `ipython python/train.py`.
Use `ipython python/train.py -- --delete` to start a clean run and remove the old checkpoints.
Please note that all commands are expected to be executed from the projects root folder.
The additional `--` before the actual flags begin is used to indicate the end of IPython flags.

The training progress can be monitored using Tensorboard.
To start Tensorboard use `tensorboard --logdir <checkpoint directory>`, it can then be viewed on
[localhost:6006](http://localhost:6006).

## Evaluation
Evaluate the current model by invoking `ipython python/evaluate.py`.
Use `ipython python/evaluate.py -- --test` to run on the test dataset, instead of the development 
set.


<!--
# vim: ts=2:sw=2:et:
-->