use super::*;

layer_documentation!(
    REGRESSION,
    "regressionLayer",
    "Create a regression output layer object.",
    "`regressionLayer(Name, Value, ...)` records regression-output metadata.",
    &["regressionLayer", "deep learning", "regression", "output layer"],
    "layer = regressionLayer('Name', 'regression'); fprintf('%s\\n', layer.Name);",
    "regression",
    "assert(strcmp(layer.Name, 'regression'));",
    "The result marks a regression output layer for a layer collection; construction does not fit a model."
);
