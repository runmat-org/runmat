use super::*;

layer_documentation!(
    CLASSIFICATION,
    "classificationLayer",
    "Create a classification output layer object.",
    "`classificationLayer(Name, Value, ...)` records classification-output metadata; Classes defaults to `auto`.",
    &["classificationLayer", "deep learning", "classification", "output layer"],
    "layer = classificationLayer('Name', 'classoutput'); fprintf('%s\\n', layer.Classes);",
    "auto",
    "assert(strcmp(layer.Classes, 'auto'));",
    "`Classes` defaults to `auto` and may be replaced by a name-value property. This constructor does not train a classifier."
);
