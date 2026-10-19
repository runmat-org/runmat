use super::*;

layer_documentation!(
    FULLY_CONNECTED,
    "fullyConnectedLayer",
    "Create a fully connected layer object.",
    "`fullyConnectedLayer(outputSize, Name, Value, ...)` requires a positive integer output size and stores common name-value properties.",
    &["fullyConnectedLayer", "deep learning", "layer", "neural network"],
    "layer = fullyConnectedLayer(10, 'Name', 'fc1'); fprintf('%g\\n', layer.OutputSize);",
    "10",
    "assert(layer.OutputSize == 10); assert(strcmp(layer.Name, 'fc1'));",
    "The result has class `nnet.cnn.layer.FullyConnectedLayer`; `OutputSize` is stored as a scalar. The R2026a positional `dim` form stores `OperationDimension`; an explicit dimension still requires compatible network execution. `Weights` and `Bias` can be supplied as name-value properties for supported forward execution."
);
