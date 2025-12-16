This application is an implementation of top-K algorithm
K here is chosen with static size = 128

Algorithm:
1. Input X
2. Set iterX = X, out = empty as iter args
3. while i < K
    - find max values and their indicies in every rows of `iterX`.
    - insert slice of the max values in `out`.
    - masked out the max-value-indicies in `iterX` by setting them to -inf.
    - increment i.


