use super::*;

layer_documentation!(
    GLOBAL_POOL,
    "globalAveragePooling1dLayer",
    "Create a global average pooling 1-D object.",
    "`globalAveragePooling1dLayer(Name, Value, ...)` records 1-D global average pooling metadata.",
    &["globalAveragePooling1dLayer", "deep learning", "pooling", "layer"],
    "layer = globalAveragePooling1dLayer('Name', 'gap'); fprintf('%s\n', layer.Name);",
    "gap",
    "assert(strcmp(layer.Name, 'gap'));",
    "The result records a global average pooling layer; construction does not pool an input sequence."
);
