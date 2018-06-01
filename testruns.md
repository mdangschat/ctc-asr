# Testruns
Listing of testruns and results.


## COSY17
| train_dir             | commit-hash                              | Br.   | Server | BS | Features | Norm.        | Units | Ep. | Layout | What was tested?                        | Loss | MED | WER |
|-----------------------|------------------------------------------|-------|--------|---:|----------|--------------|------:|----:|-------:|-----------------------------------------|-----:|----:|----:|
| `3d1r2d_global`       | 66be4305025f279f8293c1949cf4a0084c451f26 | model | cosy14 |  8 | 80 MFCC  | global       |  2048 |  20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |
| `3d1r2d_local`        | b026c59ca2b4345a658697b52311268073577940 | model | cosy15 |  8 | 80 MFCC  | local        |  2048 |  20 | 3d1r2d | DS1 w/ local MFCC normalization.        |      |     |     |
| `3d1r2d_local_scalar` | 2dc0508e222da3e5b3ba25a79c1e8574dcfda4c3 | model | cosy16 |  8 | 80 MFCC  | local scalar |  2048 |  20 | 3d1r2d | DS1 w/ local_scalar MFCC normalization. |      |     |     |
| `3d1r2d_false`        |                                          | model | cosy   |  8 | 80 MFCC  | False        |  2048 |  20 | 3d1r2d | DS1 w/o MFCC normalization.             |      |     |     |
| `3d1r2d_global_3000U` |                                          | model | cosy   |  8 | 80 MFCC  | global       |  3000 |  20 | 3d1r2d | DS1 w/ global MFCC normalization.       |      |     |     |

## GTX1080