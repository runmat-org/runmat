use super::*;

layer_documentation!(
    FEATURE_INPUT,
    "featureInputLayer",
    "Create a feature input layer object.",
    "`featureInputLayer(numFeatures, Name, Value, ...)` records a positive feature count and name-value properties. In RunMat mode, a vector of positive dimensions is also retained as layer metadata.",
    &["featureInputLayer", "deep learning", "input layer", "features"],
    "layer = featureInputLayer(4, 'Name', 'features'); fprintf('%g\\n', layer.InputSize);",
    "4",
    "assert(isequal(layer.InputSize, 4)); assert(strcmp(layer.Name, 'features'));",
    "MATLAB documents a scalar feature count. RunMat mode also accepts a vector `InputSize` as metadata; MATLAB compatibility mode rejects that extension. The current `dlnetwork` forward path supports only a scalar feature width and rejects vector metadata rather than ignoring trailing dimensions. No tensor is allocated by construction."
);
