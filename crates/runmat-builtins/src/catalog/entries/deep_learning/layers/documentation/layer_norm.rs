use super::*;

layer_documentation!(
    LAYER_NORM,
    "layerNormalizationLayer",
    "Create a layer normalization object.",
    "`layerNormalizationLayer(Name, Value, ...)` creates a normalization layer for a sequential network. Forward execution normalizes each admitted channel group, then applies its learned Scale and Offset. Epsilon defaults to `1e-5`.",
    &["layerNormalizationLayer", "deep learning", "normalization", "layer"],
    "layer = layerNormalizationLayer('Name', 'ln'); fprintf('%.5f\\n', layer.Epsilon);",
    "0.00001",
    "assert(layer.Epsilon == 1e-5); assert(strcmp(layer.Name, 'ln'));",
    "The constructor does not normalize data. Recognized learn-rate factors and initializers are retained as layer properties. Network execution supports real single and double feature, sequence, and 1-D spatial inputs; OperationDimension selects channel-only, spatial-and-channel, or batch-excluded groups. The network initializes Scale and Offset, which remain editable learnables."
);
