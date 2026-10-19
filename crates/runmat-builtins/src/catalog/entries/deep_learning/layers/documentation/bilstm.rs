use super::*;

layer_documentation!(
    BILSTM,
    "bilstmLayer",
    "Create a bidirectional LSTM layer object.",
    "`bilstmLayer(numHiddenUnits, Name, Value, ...)` requires a positive hidden-unit count and configures a bidirectional recurrent layer for network execution and training.",
    &["bilstmLayer", "deep learning", "BiLSTM", "recurrent", "layer"],
    "layer = bilstmLayer(32, 'Name', 'bilstm'); fprintf('%g\\n', layer.NumHiddenUnits);",
    "32",
    "assert(layer.NumHiddenUnits == 32); assert(strcmp(layer.Name, 'bilstm'));",
    "`NumHiddenUnits` is positive; `OutputMode` defaults to `sequence` and can be `last`. Host network execution and supervised training use the native single or double recurrent kernels. InputWeights, RecurrentWeights, and Bias learning-rate and L2 factors accept a nonnegative scalar or a 1-by-8 row vector: four gates in input, forget, cell-candidate, output order for the forward direction, then the same order for the backward direction. Explicit resident inputs are not silently gathered into this host route."
);
