# Dataset Information
`train.txt` is sorted by feature sequence length in ascending order.
`dev.txt` and `test.txt` are unsorted.

Examples that are longer than 1700 feature vectors have been removed.
Tedlium samples with labels shorter than 5 words have been removed.


### Used Datasets
* [LibriSpeech ASR Corpus](http://openslr.org/12)
* [Mozilla Common Voice](https://voice.mozilla.org/en)
* [Tatoeba](https://tatoeba.org/eng/downloads)
* [TED-LIUM v2](http://www.openslr.org/19/)
* [TIMIT](https://catalog.ldc.upenn.edu/ldc93s1)


### `train.txt` 906+ Hours
Examples shorter than 0.7 and longer than 17.0 seconds have been removed.
* `libri_speech_train.txt`
* `timit_train.txt`
* `tedlium_train.txt`
* `common_voice_train.txt`
* `tatoeba_train.txt`


### `dev.txt`
* `libri_speech_dev.txt`


### `test.txt`
* `libri_speech_test.txt`
* `common_voice_test.txt`


## Statistics
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
