# Testruns
Listing of testruns and results.


## COSY17
| train_dir                 | commit-hash                              | Branch | Server | BS | Features | Normalize    | Units | Epochs | Layout | What was tested?                        | Loss | MED | WER |
|---------------------------|------------------------------------------|--------|--------|---:|----------|--------------|------:|-------:|-------:|-----------------------------------------|-----:|----:|----:|
| `ds1_mfcc26_global`       | be39a4993db706b8b4e87d3d9b85592c631fd0fc | `ds1`  | cosy14 |  8 | MFCC     | global       |  2048 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |
| `ds1_mfcc26_local`        | 7c31219bb7f1a2023e8d933172f3d95c9dea5e0a | `ds1`  | cosy17 |  8 | MFCC     | local        |  2048 |     20 | 3d1r2d | DS1 /w local MFCC normalization.        |      |     |     |
| `ds1_mfcc26_local_scalar` | c9abae42760fb01339e3c2fd46ce22cae6af0781 | `ds1`  | cosy15 |  8 | MFCC     | local scalar |  2048 |     20 | 3d1r2d | DS1 /w local_scalar MFCC normalization. |      |     |     |
| `ds1_mfcc26_global_3000U` |                                          | `ds1`  |        |  8 | MFCC     | global       |  3000 |     20 | 3d1r2d | DS1 /w global MFCC normalization.       |      |     |     |

## GTX1080