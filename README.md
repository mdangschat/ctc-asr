# Speech Recognition System


## Installation (Only Notes ATM)

### Required Libraries
```shell
pacman -S tr
```

## Datasets
### Prepare Training Data
```shell
cd project_root/data/

cat *_train.txt > train.txt
cat *_text.txt > text.txt
cat *_validate.txt > validate.txt

# Alternatively, only use the desired datasets.
cat libri_speech_train.txt tedlium_train.txt > train.txt
```


<!--
# vim: ts=2:sw=2:et:
-->
