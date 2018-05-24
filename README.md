# Single Stage Speech Recognition System using Connectionist Temporal Classification

Automatic Speech Recognition (ASR) system implementation inspired by Baidu's
[Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567) paper.

![TensorFlow Network Graph](images/graph.png)

## Installation (incomplete)
```bash
git clone https://vcs.zwuenf.org/mdangschat/speech.git
```

### Install Required Libraries
#### Arch Linux

*TODO: Add dependenies: sox, libsox-fmt-mp3*
```sh
# This list is incomplete.
pacaur -S tr

# Install TensorFlow
pacaur -S python-tensorflow-opt-cuda tensorbaord
```


## Prepare Datasets
```sh
cd <project_root>/data/

cat *_train.txt > train.txt
cat *_text.txt > text.txt
cat *_dev.txt > dev.txt

# Alternatively, only use the desired datasets.
cat libri_speech_train.txt tedlium_train.txt > train.txt
```


## Compile and Install with WarpCTC-Support
### Compile TensorFlow
```sh
# Tensorflow
git clone https://github.com/tensorflow/tensorflow
cd tensorflow

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
and [TensorFlow binding for WarpCTC](https://github.com/baidu-research/warp-ctc/tree/master/tensorflow_binding) 
for further informations.


## Training
Start training by invoking `python/train.py`.


## Evaluation
Evaluate the current model by invoking `python/evaluate.py`.
 

<!--
# vim: ts=2:sw=2:et:
-->