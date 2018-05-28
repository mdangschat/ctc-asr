# Testruns
Listing of testruns and results.


## COSY17
| train_dir                 | commit-hash                              | Branch | Server | BS | Features | Normalize    | Units | Epochs | Layout | What was tested?                        | Loss | MED | WER |
|---------------------------|------------------------------------------|--------|--------|---:|----------|--------------|------:|-------:|-------:|-----------------------------------------|-----:|----:|----:|
| `3d1r2d_global`           | 733daf56644b1f959873ef61ff1580f6fcac91eb | model  | cosy14 |  8 | 80 MFCC  | global       |  2048 |     20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |
| `3d1r2d_local`            | f5f061a2edb7d5e4af43f75b19fe068d91043541 | model  | cosy15 |  8 | 80 MFCC  | local        |  2048 |     20 | 3d1r2d | DS1 w/ local MFCC normalization.        |      |     |     |
| `3d1r2d_local_scalar`     | d8c39ef3034a05cb40cec6a165e5d1732fc58781 | model  | cosy16 |  8 | 80 MFCC  | local scalar |  2048 |     20 | 3d1r2d | DS1 w/ local_scalar MFCC normalization. |      |     |     |
| `ds1_mfcc26_false`        |                                          | model  | cosy   |  8 | 80 MFCC  | False        |  2048 |     20 | 3d1r2d | DS1 w/o MFCC normalization.             |      |     |     |
| `ds1_mfcc26_global_3000U` |                                          | model  | cosy   |  8 | 80 MFCC  | global       |  3000 |     20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |

## GTX1080