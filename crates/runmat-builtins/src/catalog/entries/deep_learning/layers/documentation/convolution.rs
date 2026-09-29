use super::*;

layer_documentation!(
    CONVOLUTION,
    "convolution1dLayer",
    "Create a one-dimensional convolution layer object.",
    "`convolution1dLayer(filterSize, numFilters, Name, Value, ...)` requires two positive integer sizes and stores convolution metadata.",
    &["convolution1dLayer", "deep learning", "convolution", "layer"],
    "layer = convolution1dLayer(3, 16, 'Padding', 'causal', 'Name', 'conv'); fprintf('%g\n', layer.NumFilters);",
    "16",
    "assert(layer.FilterSize == 3); assert(layer.NumFilters == 16); assert(strcmp(layer.Padding, 'causal'));",
    "`FilterSize` and `NumFilters` are required positive integers. `Stride` defaults to one, `Padding` to `same`, and both can be overridden by name-value properties."
);
