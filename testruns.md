# Testruns
Listing of testruns and results.


## COSY
| train_dir             | Branch                 | Server | BS | Features | Norm.        | Units | Ep. | Layout | What was tested?                       | Loss | MED | WER |
|-----------------------|------------------------|--------|---:|----------|--------------|------:|----:|-------:|----------------------------------------|-----:|----:|----:|
| `3d1r2d_global`       | `run_ds1_global`       | cosy14 |  8 | 80 Mel   | global       |  2048 |  20 | 3d1r2d | DS1 w/ global Mel normalization.       |      |     |     |
| `3d1r2d_local`        | `run_ds1_local`        | cosy15 |  8 | 80 Mel   | local        |  2048 |  20 | 3d1r2d | DS1 w/ local Mel normalization.        |      |     |     |
| `3d1r2d_local_scalar` | `run_ds1_local_scalar` | cosy16 |  8 | 80 Mel   | local scalar |  2048 |  20 | 3d1r2d | DS1 w/ local_scalar Mel normalization. |      |     |     |
| `3d1r2d_false`        |                        | cosy   |  8 | 80 Mel   | none         |  2048 |  20 | 3d1r2d | DS1 w/o Mel normalization.             |      |     |     |
| `3d1r2d_global_3000U` |                        | cosy   |  8 | 80 Mel   | global       |  3000 |  20 | 3d1r2d | DS1 w/ global Mel normalization.       |      |     |     |


##### Note: That runs on the COSY servers did not use the complete dataset.
* train: timit, tedlium, libri_speech, common_voice
* test: libri_speech, common_voice
* dev: libri_speech


## GTX1080