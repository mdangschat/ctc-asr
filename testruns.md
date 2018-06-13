# Testruns
Listing of testruns and results.


## COSY
| train_dir             | Branch                 | Server | BS | Features | Norm.        | Units | Ep. | Layout | What was tested?                       |   Loss |   MED |   WER |
|-----------------------|------------------------|--------|---:|----------|--------------|------:|----:|-------:|----------------------------------------|-------:|------:|------:|
| `3d1r2d_global`       | `run_ds1_global`       | cosy14 |  8 | 80 Mel   | global       |  2048 |  20 | 3d1r2d | DS1 w/ global Mel normalization.       | 30.594 | 0.113 | 0.319 |
| `3d1r2d_local`        | `run_ds1_local`        | cosy15 |  8 | 80 Mel   | local        |  2048 |  20 | 3d1r2d | DS1 w/ local Mel normalization.        | 29.022 | 0.107 | 0.309 |
| `3d1r2d_local_scalar` | `run_ds1_local_scalar` | cosy16 |  8 | 80 Mel   | local scalar |  2048 |  20 | 3d1r2d | DS1 w/ local_scalar Mel normalization. | 31.882 | 0.114 | 0.321 |
| `3d1r2d_none`         | `run_ds1_none`         | cosy14 |  8 | 80 Mel   | none         |  2048 |  20 | 3d1r2d | DS1 w/o Mel normalization.             |        |       |       |
| `3d1r2d_mfcc_local`   |                        | cosy15 |  8 | 80 MFCC  | local        |  2048 |  20 | 3d1r2d | DS1 w/ local MFCC normalization.       |        |       |       |
| `3d1r2d_local_3000u`  |                        | cosy16 |  8 | 80 Mel   | local        |  3000 |  20 | 3d1r2d | DS1 w/ global Mel normalization.       |        |       |       |


##### Note: That runs on the COSY servers did not use the complete dataset.
* train: timit, tedlium, libri_speech, common_voice
* test: libri_speech, common_voice
* dev: libri_speech


## GTX1080