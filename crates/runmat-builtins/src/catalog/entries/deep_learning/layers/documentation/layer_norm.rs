use super::*;

layer_documentation!(
    LAYER_NORM,
    "layerNormalizationLayer",
    "Create a layer normalization object.",
    "`layerNormalizationLayer(Name, Value, ...)` records normalization metadata; Epsilon defaults to `1e-5`.",
    &["layerNormalizationLayer", "deep learning", "normalization", "layer"],
    "layer = layerNormalizationLayer('Name', 'ln'); fprintf('%.5f\n', layer.Epsilon);",
    "0.00001",
    "assert(layer.Epsilon == 1e-5); assert(strcmp(layer.Name, 'ln'));",
    "`Epsilon` defaults to `1e-5`; the constructor stores options without normalizing data."
);
