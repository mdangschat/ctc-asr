# End-to-End Speech Recognition System Using Connectionist Temporal Classification
Automatic speech recognition (ASR) system implementation inspired by Baidu's
[Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567) and
[Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595) 
papers.
The system is trained on a combined corpus, containing 900+ hours.
It achieves a word error rate (WER) of 12.6% on the test dataset, without the use of an external
language model.

![Deep Speech 1 Network Architecture](images/ds1-network-architecture.png)
![Modified Network Architecture](images/ds2-network-architecture.png)


## Contents
* [Installation](#installation)
  * [Arch Linux](#arch-linux)
  * [Ubuntu](#ubuntu-1604)
* [Datasets](#datasets)
* [Configutation](#configuration)
* [Training](#training)
* [Evaluation](#evaluation)
* [License](#license)


## Installation
The system was tested on Arch Linux and Ubuntu 16.04, with Python version 3.5+ and the 1.9.0 version
of [TensorFlow](https://www.tensorflow.org/). It is highly recommended to use TensorFlow with GPU
support for training.

Be aware that the [requirements.txt](requirements.txt) lists `tensorflow` as dependency, if you
install TensorFlow through [pip](https://pypi.org/project/pip/) consider removing it and 
install `tensorflow-gpu` manually.
Based on my experience it is worth the effort to 
[build TensorFlow from source](https://www.tensorflow.org/install/source).


### Arch Linux
```sh
# Install dependencies
pacman -S tr sox python-tensorflow-opt-cuda tensorbaord

# Clone reposetory and install Python depdendencies
git clone https://vcs.zwuenf.org/mdangschat/speech.git
cd speech
pip install -r requirements.txt
```

### Ubuntu 16.04
```sh
sudo apt install python3-tk sox libsox-fmt-all

# Clone reposetory and install Python depdendencies
git clone https://vcs.zwuenf.org/mdangschat/speech.git
cd speech
pip3 install -r requirements.txt
```


## Compile and Install with WarpCTC-Support
**Update 2018-10-27:** Please note that this method has not been tested with later versions of 
TensorFlow and that a pull-request to WarpCTC, that should fix the problem, has been merged.


### Compile TensorFlow
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


### Compile WarpCTC
```sh
# Back to base folder
cd ..

# Set environment variables.
export CUDA_HOME="/usr/local/cuda"
export TENSORFLOW_SRC_PATH="/home/marc/workspace/tensorflow"
export WARP_CTC_PATH="/home/marc/workspace/warp-ctc/build"

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

Reference [Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources) 
and 
[TensorFlow binding for WarpCTC](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding) 
for further information.


## Configuration
The network architecture and training parameters can be configured by adding the appropriate flags
or by directly editing the [params.py](asr/params.py) configuration file.


## Training
Start training by invoking `ipython asr/train.py`.
Use `ipython asr/train.py -- --delete` to start a clean run and remove the old checkpoints.
Please note that all commands are expected to be executed from the projects root folder.
The additional `--` before the actual flags begin is used to indicate the end of IPython flags.


## Evaluation
Evaluate the current model by invoking `ipython asr/evaluate.py`.
Use `ipython asr/evaluate.py -- --test` to run on the test dataset, instead of the development one.


## License
Copyright (c) 2018 Marc Dangschat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



<!--
# vim: ts=2:sw=2:et:
-->