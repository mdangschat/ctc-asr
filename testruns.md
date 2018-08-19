## Testruns
Listing of testruns and results.


### COSY (Reduced Dataset)
| train_dir             | Server | BS | Input   | Norm.        | Units | Ep | Layout | Loss   | MED   |   WER | Notes          |
|-----------------------|--------|---:|---------|--------------|------:|---:|-------:|-------:|------:|------:|----------------|
| `3d1r2d_global`       | cosy14 |  8 | 80 Mel  | global       |  2048 | 20 | 3d1r2d | 30.594 | 0.113 | 0.319 |                |
| `3d1r2d_local`        | cosy15 |  8 | 80 Mel  | local        |  2048 | 20 | 3d1r2d | 29.022 | 0.107 | 0.309 |                |
| `3d1r2d_local_scalar` | cosy16 |  8 | 80 Mel  | local scalar |  2048 | 20 | 3d1r2d | 31.882 | 0.114 | 0.321 |                |
| `3d1r2d_none`         | cosy14 |  8 | 80 Mel  | none         |  2048 | 20 | 3d1r2d | 29.604 | 0.112 | 0.317 |                |
| `3d1r2d_mfcc_local`   | cosy15 |  8 | 80 MFCC | local        |  2048 | 20 | 3d1r2d | 24.633 | 0.088 | 0.255 |                |
| `3d1r2d_local_3000u`  | cosy16 |  8 | 80 Mel  | local        |  3000 | 20 | 3d1r2d | 34.556 | 0.102 | 0.290 |                |


#### Reduced Dataset
Note that runs marked with *Reduced Dataset* did not use the complete dataset.
* train: timit, tedlium, libri_speech, common_voice
* test: libri_speech, common_voice
* dev: libri_speech


### COSY
| train_dir                      | Server | BS | Input   | Norm.        | Units | Ep | Layout | Loss   | MED   | WER   | Notes          |
|--------------------------------|--------|---:|---------|--------------|------:|---:|-------:|-------:|------:|------:|----------------|
| `3d1r2d_global_mfcc_full`      | cosy14 |  8 | 80 MFCC | global       |  2048 | 20 | 3d1r2d | 25.606 | 0.106 | 0.304 |                |
| `3d2r2d_local_mfcc_full`       | cosy15 |  8 | 80 MFCC | local        |  2048 | 16 | 3d2r2d | 18.988 | 0.074 | 0.211 | Stopped early. |
| `3d1r2d_global_mel_full`       | cosy14 |  8 | 80 Mel  | global       |  2048 | 14 | 3d1r2d | 31.399 | 0.131 | 0.371 | Stopped early  |
| `3d1r2d_local_mel_full`        | cosy15 |  8 | 80 Mel  | local        |  2048 | 15 | 3d1r2d | 29.520 | 0.125 | 0.354 | Stopped early. |
| `3d1r2d_local_scalar_mel_full` | cosy16 |  8 | 80 Mel  | local scalar |  2048 | 15 | 3d1r2d | 31.669 | 0.132 | 0.373 | Stopped early. |
| `3d1r2d_none_mel_full`         | cosy17 |  8 | 80 Mel  | none         |  2048 | 16 | 3d1r2d | 32.006 | 0.135 | 0.376 | Stopped early. |
| `3c1r2d_mel_local_full`        | cosy17 |  8 | 80 Mel  | local        |  2048 |    | 3c1r2d |        |       |       | Stopped early  |
| `3c1r2d_mel_localscalar_full`  | cosy14 |  8 | 80 Mel  | local scalar |  2048 |  9 | 3c1r2d | 23.579 | 0.090 | 0.256 | Stopped early. |
| `3c1r2d_mel_global_full`       | cosy15 |  8 | 80 Mel  | global       |  2048 |  9 | 3c1r2d | 24.059 | 0.094 | 0.267 | Stopped early. |
| `3c1r2d_mel_none_full`         | cosy16 |  8 | 80 Mel  | none         |  2048 |  9 | 3c1r2d | 26.979 | 0.106 | 0.292 | Stopped early. |


### FB02TIITs04; V100 32GB
| train_dir                    | BS | Input   | Norm. | Units | Ep | Layout | Loss  | MED   | WER    | Notes                       |
|------------------------------|---:|---------|-------|------:|---:|-------:|------:|------:|-------:|-----------------------------|
| `3c1r2d_mel_local_full`      |  8 | 80 Mel  | local |  2048 | 20 | 3c4r2d | 25.43 | 0.083 | 0.2412 |                             |
| `3c3r2d_mel_local`           |  8 | 80 Mel  | local |  2048 | 11 | 3c3r2d | 17.32 | 0.062 | 0.1762 | Stopped early.              |
| `3c4r2d_mel_local_full_lstm` |  8 | 80 Mel  | local |  2048 |  5 | 3c4r2d | 11.849| 0.045 | 0.1264 | LSTM cells.                 |
| `3c5r2d_mel_local_full`      |  8 | 80 Mel  | local |  2048 |  9 | 3c5r2d | 13.26 | 0.044 | 0.1292 | LSTM cells. Server crashed. |


### FB11-NX-T02; 2xV100 16GB
| train_dir                     | BS | Input   | Norm. | Units | Ep | Layout | Loss  | MED   | WER    | Notes                 |
|-------------------------------|---:|---------|-------|------:|---:|-------:|------:|------:|-------:|-----------------------|
| `3c5r2d_mel_local_full_bs16`  | 16 | 80 Mel  | local |  2048 | 10 | 3c5r2d | 14.02 | 0.057 | 0.1583 | Stopped early.        |
| `3c5r2d_mfcc_local_full_bs16` | 16 | 80 MFCC | local |  2048 | 17 | 3c5r2d | 19.63 | 0.081 | 0.2207 | Tanh RNN.             |
| `3c4r2d_mfcc_local_bs16_relu` | 16 | 80 MFCC | local |  2048 | 16 | 3c4r2d | 20.45 | 0.081 | 0.2273 | ReLU RNN. HDD full.   |


### FB11-NX-T01; 1xV100 16GB
| train_dir                     | BS | Input   | Norm. | Units | Ep | Layout | Loss  | MED   | WER    | Notes                     |
|-------------------------------|---:|---------|-------|------:|---:|-------:|------:|------:|-------:|---------------------------|
| `3c4r2d_mfcc_local_bs16_gru`  | 16 | 80 MFCC | local |  2048 | 10 | 3c4r2d | 16.78 | 0.067 | 0.1913 | GRU cells.                |
| `3c3r2d_mel_local_bs16_tanh`  | 16 | 80 Mel  | local |  2048 | 15 | 3c3r2d | 17.72 | 0.072 | 0.2059 | ReLU cells, despite name. |
