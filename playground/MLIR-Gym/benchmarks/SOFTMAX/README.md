This is application is an implementation of naive softmax kernel.

Algorithm:
1. Row reduction max (rowmax)
2. Broadcast rowmax + subtraction: X-rowmax
3. Row reduction sum (rowsum) over exp(X-rowmax)
4. Broadcast rowsum + division: X-rowmax / rowsum
