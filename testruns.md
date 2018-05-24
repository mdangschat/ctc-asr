# Testruns
Listing of testruns and results.


## COSY17
| train_dir                 | commit-sha | Branch | BS | Features | Normalize    | Units | Epochs | Layout | What was tested?                        | Loss | MED | WER |
|---------------------------|------------|--------|---:|----------|--------------|------:|-------:|--------|-----------------------------------------|-----:|----:|----:|
| `ds1_mfcc26_global`       | 6969311d4a | `ds1`  |  8 |     MFCC |       global |  2048 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |
| `ds1_mfcc26_local`        |            | `ds1`  |  8 |     MFCC |        local |  2048 |     20 | 3d1r2d | DS1 /w local MFCC normalization.        |      |     |     |
| `ds1_mfcc26_local_scalar` |            | `ds1`  |  8 |     MFCC | local scalar |  2048 |     20 | 3d1r2d | DS1 /w local_scalar MFCC normalization. |      |     |     |
| `ds1_mfcc26_global_3000U` |            | `ds1`  |  8 |     MFCC |       global |  3000 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |


## GTX1080