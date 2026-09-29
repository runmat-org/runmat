use super::*;

layer_documentation!(
    RELU,
    "reluLayer",
    "Create a ReLU activation layer object.",
    "`reluLayer(Name, Value, ...)` stores ReLU activation metadata.",
    &["reluLayer", "deep learning", "activation", "ReLU"],
    "layer = reluLayer('Name', 'relu1'); fprintf('%s\\n', layer.Name);",
    "relu1",
    "assert(strcmp(layer.Name, 'relu1'));",
    "The object records a ReLU activation layer and its optional properties."
);
