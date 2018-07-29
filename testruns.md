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
| `3d1r2d_global_mel_full`       | cosy14 |  8 | 80 Mel  | global       |  2048 | 20 | 3d1r2d |        |       |       |                |
| `3d1r2d_local_mel_full`        | cosy15 |  8 | 80 Mel  | local        |  2048 | 20 | 3d1r2d |        |       |       |                |
| `3d1r2d_local_scalar_mel_full` | cosy16 |  8 | 80 Mel  | local scalar |  2048 | 20 | 3d1r2d |        |       |       |                |
| `3d1r2d_none_mel_full`         | cosy17 |  8 | 80 Mel  | none         |  2048 | 20 | 3d1r2d |        |       |       |                |


### FB02TIITs04; V100 32GB
| train_dir               | BS | Input   | Norm. | Units | Ep | Layout | Loss | MED | WER | Notes                 |
|-------------------------|---:|-------- |-------|------:|---:|-------:|-----:|----:|----:|-----------------------|
| `3c3r2d_mel_local`      |  8 | 80 Mel  | local |  2048 | 11 | 3c3r2d |      |     |     | Stopped early.        |
| `3c4r2d_mel_local_full` |  8 | 80 Mel  | local |  2048 |    | 3c4r2d |      |     |     |                       |
| `3c7r2d_mel_local_full` |  8 | 80 Mel  | local |  2048 |    | 3c7r2d |      |     |     |                       |


### FB11-NX-T02; 2xV100 16GB
| train_dir                    | BS | Input   | Norm. | Units | Ep | Layout | Loss | MED | WER | Notes                 |
|------------------------------|---:|---------|-------|------:|---:|-------:|-----:|----:|----:|-----------------------|
| `3c5r2d_mel_local_full_bs16` | 16 | 80 Mel  | local |  2048 | 11 | 3c5r2d |      |     |     | Stopped early.        |



