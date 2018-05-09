# Speech Recognition System


## Installation (Only Notes ATM)

### Required Libraries
```sh
pacman -S tr
```

## Datasets
### Prepare Training Data
```sh
cd project_root/data/

cat *_train.txt > train.txt
cat *_text.txt > text.txt
cat *_validate.txt > validate.txt

# Alternatively, only use the desired datasets.
cat libri_speech_train.txt tedlium_train.txt > train.txt
```


## Install with WarpCTC support

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

# Warp CTC
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

<!--
# vim: ts=2:sw=2:et:
-->
