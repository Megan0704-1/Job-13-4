This application is an implementation of RMS Normalization kernel.

Algorithm:
1. Calculate sum of squared root for each row -> row reduce squared sum (rrss)
2. Calculate mean from each rrss -> rrss / N (mrrss)
3. Calculate inverse term -> 1 / sqrt (mrrss + ep)
4. Calculate Y -> Y = X * (inverse term) * gamma

Note. RMS norm is scale invariance but not shift invariant.
