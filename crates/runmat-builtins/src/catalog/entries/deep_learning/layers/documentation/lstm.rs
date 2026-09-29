use super::*;

layer_documentation!(
    LSTM,
    "lstmLayer",
    "Create an LSTM layer object.",
    "`lstmLayer(numHiddenUnits, Name, Value, ...)` requires a positive hidden-unit count and records recurrent layer metadata.",
    &["lstmLayer", "deep learning", "LSTM", "recurrent", "layer"],
    "layer = lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'lstm'); fprintf('%g\n', layer.NumHiddenUnits);",
    "64",
    "assert(layer.NumHiddenUnits == 64); assert(strcmp(layer.OutputMode, 'sequence'));",
    "`NumHiddenUnits` stores a positive integer; `OutputMode` defaults to `sequence` and can be overridden with `last`. This implementation stores recurrent metadata but does not execute an LSTM during `forward`."
);
