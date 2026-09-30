use super::*;

layer_documentation!(
    LSTM,
    "lstmLayer",
    "Create an LSTM layer object.",
    "`lstmLayer(numHiddenUnits, Name, Value, ...)` requires a positive hidden-unit count and configures a recurrent layer for network execution and training.",
    &["lstmLayer", "deep learning", "LSTM", "recurrent", "layer"],
    "layer = lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'lstm'); fprintf('%g\\n', layer.NumHiddenUnits);",
    "64",
    "assert(layer.NumHiddenUnits == 64); assert(strcmp(layer.OutputMode, 'sequence'));",
    "`NumHiddenUnits` is positive; `OutputMode` defaults to `sequence` and can be `last`. Host network execution and supervised training use the native single or double recurrent kernels. InputWeights, RecurrentWeights, and Bias learning-rate and L2 factors accept a nonnegative scalar or a 1-by-4 row vector in input, forget, cell-candidate, output gate order. Explicit resident inputs are not silently gathered into this host route."
);
