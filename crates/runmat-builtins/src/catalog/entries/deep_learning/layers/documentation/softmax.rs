use super::*;

layer_documentation!(
    SOFTMAX,
    "softmaxLayer",
    "Create a softmax layer object.",
    "`softmaxLayer(Name, Value, ...)` stores softmax layer metadata.",
    &["softmaxLayer", "deep learning", "softmax", "layer"],
    "layer = softmaxLayer('Name', 'prob'); fprintf('%s\\n', layer.Name);",
    "prob",
    "assert(strcmp(layer.Name, 'prob'));",
    "The object records the softmax layer kind and optional metadata."
);
