# Dataset Information


## General Information
[LibriSpeech ASR Corpus](http://openslr.org/12), [TED-LIUM v2](http://www.openslr.org/19/), 
[Mozilla Common Voice](https://voice.mozilla.org/en), and 
[TIMIT](https://catalog.ldc.upenn.edu/ldc93s1) have been used.
`train.txt` is sorted by feature sequence length in ascending order, while `validate.txt` and 
`test.txt` is unsorted.


## `train.txt`
Examples that are longer than 2500 feature vectors have been removed.

* `libri_speech_train.txt`
* `timit_complete.txt`
  * `timit_train.txt`
  * `timit_test.txt`
* `tedlium_train.txt`
* `common_voice_train.txt`


## `validate.txt`
* `libri_speech_validate.txt`


## `test.txt`
* `libri_speech_test.txt`
* `common_voice_test.txt`