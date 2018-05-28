# Testruns
Listing of testruns and results.


## COSY17
| train_dir             | commit-hash                              | Br.   | Server | BS | Features | Norm.        | Units | Ep. | Layout | What was tested?                        | Loss | MED | WER |
|-----------------------|------------------------------------------|-------|--------|---:|----------|--------------|------:|----:|-------:|-----------------------------------------|-----:|----:|----:|
| `3d1r2d_global`       | e0796edd1d4bfff73159e5956dd575d20d1c89a5 | model | cosy14 |  8 | 80 MFCC  | global       |  2048 |  20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |
| `3d1r2d_local`        | bd95bbb0e7fc2f18f2f0d3ccbaf378fc5fd69916 | model | cosy15 |  8 | 80 MFCC  | local        |  2048 |  20 | 3d1r2d | DS1 w/ local MFCC normalization.        |      |     |     |
| `3d1r2d_local_scalar` | 24aef61473d6cefd679320b1043f69ae369a1b8b | model | cosy16 |  8 | 80 MFCC  | local scalar |  2048 |  20 | 3d1r2d | DS1 w/ local_scalar MFCC normalization. |      |     |     |
| `3d1r2d_false`        |                                          | model | cosy   |  8 | 80 MFCC  | False        |  2048 |  20 | 3d1r2d | DS1 w/o MFCC normalization.             |      |     |     |
| `3d1r2d_global_3000U` |                                          | model | cosy   |  8 | 80 MFCC  | global       |  3000 |  20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |

## GTX1080