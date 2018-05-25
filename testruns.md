# Testruns
Listing of testruns and results.


## COSY17
| train_dir                 | commit-hash                              | Branch | Server | BS | Features | Normalize    | Units | Epochs | Layout | What was tested?                        | Loss | MED | WER |
|---------------------------|------------------------------------------|--------|--------|---:|----------|--------------|------:|-------:|-------:|-----------------------------------------|-----:|----:|----:|
| `ds1_mfcc26_global`       | 733daf56644b1f959873ef61ff1580f6fcac91eb | `ds1`  | cosy14 |  8 | MFCC     | global       |  2048 |     20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |
| `ds1_mfcc26_local`        | f5f061a2edb7d5e4af43f75b19fe068d91043541 | `ds1`  | cosy15 |  8 | MFCC     | local        |  2048 |     20 | 3d1r2d | DS1 w/ local MFCC normalization.        |      |     |     |
| `ds1_mfcc26_local_scalar` | d8c39ef3034a05cb40cec6a165e5d1732fc58781 | `ds1`  | cosy16 |  8 | MFCC     | local scalar |  2048 |     20 | 3d1r2d | DS1 w/ local_scalar MFCC normalization. |      |     |     |
| `ds1_mfcc26_false`        |                                          | `ds1`  | cosy   |  8 | MFCC     | False        |  2048 |     20 | 3d1r2d | DS1 w/o MFCC normalization.             |      |     |     |
| `ds1_mfcc26_global_3000U` |                                          | `ds1`  |        |  8 | MFCC     | global       |  3000 |     20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |

## GTX1080