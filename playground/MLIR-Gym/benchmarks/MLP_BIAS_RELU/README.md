1. This example is matmul + bias add + relu
2. 2 pipelines differ at the ordering of fusion, one before bufferization, one after
3. the result of the former one is expected to have fewer load-store and faster speed than the later
