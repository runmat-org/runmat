use super::*;

layer_documentation!(
    SEQUENCE_INPUT,
    "sequenceInputLayer",
    "Create a sequence input layer object.",
    "`sequenceInputLayer(inputSize, Name, Value, ...)` records positive sequence-input dimensions and name-value properties.",
    &["sequenceInputLayer", "deep learning", "sequence", "input layer"],
    "layer = sequenceInputLayer(12, 'Name', 'tokens'); fprintf('%g\\n', layer.InputSize);",
    "12",
    "assert(isequal(layer.InputSize, 12)); assert(strcmp(layer.Name, 'tokens'));",
    "`InputSize` is stored as a row vector of positive dimensions; sequence data is supplied later to a model."
);
