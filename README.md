# Bug

- sample/sample.cpp 需更改，不能编译
- gat的模型nhead=1的时候实现不了，edge_softmax算子就报cudaErrorIllegalAddress的错
