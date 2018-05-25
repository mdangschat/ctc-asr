# Testruns
Listing of testruns and results.


## COSY17
| train_dir                 | commit-hash                              | Branch | Server | BS | Features | Normalize    | Units | Epochs | Layout | What was tested?                        | Loss | MED | WER |
|---------------------------|------------------------------------------|--------|--------|---:|----------|--------------|------:|-------:|-------:|-----------------------------------------|-----:|----:|----:|
| `ds1_mfcc26_global`       | f621cca25ce11d41ab3438294c2445adda220297 | `ds1`  | cosy14 |  8 | MFCC     | global       |  2048 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |
| `ds1_mfcc26_local`        | a592893bdddd2dfd676f62f45cc62cdbdacf6025 | `ds1`  | cosy15 |  8 | MFCC     | local        |  2048 |     20 | 3d1r2d | DS1 /w local MFCC normalization.        |      |     |     |
| `ds1_mfcc26_local_scalar` | 5a5d212294145e17e6ccbf2400b748732d49ab37 | `ds1`  | cosy16 |  8 | MFCC     | local scalar |  2048 |     20 | 3d1r2d | DS1 /w local_scalar MFCC normalization. |      |     |     |
| `ds1_mfcc26_global_3000U` |                                          | `ds1`  |        |  8 | MFCC     | global       |  3000 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |

## GTX1080