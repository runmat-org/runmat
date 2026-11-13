use super::*;

layer_documentation!(
    GLOBAL_POOL,
    "globalAveragePooling1dLayer",
    "Create a global average pooling 1-D object.",
    "`globalAveragePooling1dLayer(Name, Value, ...)` records 1-D global average pooling metadata.",
    &["globalAveragePooling1dLayer", "deep learning", "pooling", "layer"],
    "layer = globalAveragePooling1dLayer('Name', 'gap'); fprintf('%s\\n', layer.Name);",
    "gap",
    "assert(strcmp(layer.Name, 'gap'));",
    "Construction records a layer without pooling an input. Pass a network containing this layer to forward or predict to average its sequence or spatial axis. Execution preserves the predictor's single/double precision and updates the output's semantic axes. Supported provider routes keep the data resident."
);
