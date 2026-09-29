use super::*;

layer_documentation!(
    BILSTM,
    "bilstmLayer",
    "Create a bidirectional LSTM layer object.",
    "`bilstmLayer(numHiddenUnits, Name, Value, ...)` requires a positive hidden-unit count and stores bidirectional recurrent layer metadata.",
    &["bilstmLayer", "deep learning", "BiLSTM", "recurrent", "layer"],
    "layer = bilstmLayer(32, 'Name', 'bilstm'); fprintf('%g\\n', layer.NumHiddenUnits);",
    "32",
    "assert(layer.NumHiddenUnits == 32); assert(strcmp(layer.Name, 'bilstm'));",
    "`NumHiddenUnits` is positive; `OutputMode` defaults to `sequence` and can be overridden with `last`. This implementation stores recurrent metadata but does not execute a BiLSTM during `forward`."
);
