This application is an implementation of Layer Normalization kernel.

Algorithm:
1. Calculate sum for each row -> row reduce sum
2. Calculate mean from each row -> row reduce sum / N
3. Calculate squared deviation for each row -> sum of (X - mean)^2
4. Calculate vraiance -> squared deviations / N
5. Calculate inverse standard deviation -> 1 / sqrt(std + epsilon)
6. Calculate Y -> Y = (X - mean) * inverse std * gamma + bias

Note. LMS norm is shift invariant and scale invariant, but involves more ops.
