# Testruns
Listing of testruns and results.


## COSY17
| train_dir                 | commit-hash                              | Branch | Server | BS | Features | Normalize    | Units | Epochs | Layout | What was tested?                        | Loss | MED | WER |
|---------------------------|------------------------------------------|--------|--------|---:|----------|--------------|------:|-------:|-------:|-----------------------------------------|-----:|----:|----:|
| `ds1_mfcc26_global`       | 45481eb09ca3dfc3f9d78316d4f3b5e169313637 | `ds1`  |        |  8 | MFCC     | global       |  2048 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |
| `ds1_mfcc26_local`        | 7c31219bb7f1a2023e8d933172f3d95c9dea5e0a | `ds1`  | cosy17 |  8 | MFCC     | local        |  2048 |     20 | 3d1r2d | DS1 /w local MFCC normalization.        |      |     |     |
| `ds1_mfcc26_local_scalar` | 7690d8e99e8b8ad5423e2d6ba7ada8f09e80e213 | `ds1`  | cosy14 |  8 | MFCC     | local scalar |  2048 |     20 | 3d1r2d | DS1 /w local_scalar MFCC normalization. |      |     |     |
| `ds1_mfcc26_global_3000U` |                                          | `ds1`  |        |  8 | MFCC     | global       |  3000 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |

## GTX1080