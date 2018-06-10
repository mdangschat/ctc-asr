# Dataset Information


## General Information
Used datasets:
* [LibriSpeech ASR Corpus](http://openslr.org/12)
* [TED-LIUM v2](http://www.openslr.org/19/)
* [Mozilla Common Voice](https://voice.mozilla.org/en)
* [TIMIT](https://catalog.ldc.upenn.edu/ldc93s1)
* [Tatoeba](https://tatoeba.org/eng/downloads)

`train.txt` is sorted by feature sequence length in ascending order.
`dev.txt` and `test.txt` are unsorted.

Examples that are longer than 1700 feature vectors have been removed.
Tedlium samples with labels shorter than 5 words have been removed.


## `train.txt`

* `libri_speech_train.txt`
* `timit_complete.txt`
  * `timit_train.txt`
  * `timit_test.txt`
* `tedlium_train.txt`
* `common_voice_train.txt`
* `tatoeba_train.txt`

## `dev.txt`
* `libri_speech_dev.txt`


## `test.txt`
* `libri_speech_test.txt`
* `common_voice_test.txt`

