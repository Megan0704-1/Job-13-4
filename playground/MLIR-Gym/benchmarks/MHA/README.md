This application is an implementation of multi-headed attention.

Input:
- Q: [B, H, S_q, D_k]
- K: [B, H, S_k, D_k]
- V: [B, H, S_k, D_v]

Kernel assumes X is already projected and reshaped.

Algorithm:
1. Q @ K^T, reduce over D_k
2. element-wise multiplication on 1/sqrt(D_k)
3. rowmax
4. broadcast and subtract
5. rowsum
6. broadcast and division (exp(X')/sum(exp(X'))), output P
7. P @ V, reduce over S_k, get O = [B, H, S_q, D_v]
