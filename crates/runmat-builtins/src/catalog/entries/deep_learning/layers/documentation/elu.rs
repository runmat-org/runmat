use super::*;

layer_documentation!(
    ELU,
    "eluLayer",
    "Create an ELU activation layer object.",
    "`eluLayer(alpha, Name, Value, ...)` stores a finite activation scale; omitted alpha defaults to one.",
    &["eluLayer", "deep learning", "activation", "ELU"],
    "layer = eluLayer(0.5, 'Name', 'elu1'); fprintf('%g\\n', layer.Alpha);",
    "0.5",
    "assert(layer.Alpha == 0.5); assert(strcmp(layer.Name, 'elu1'));",
    "`Alpha` is a finite numeric scalar. A resident alpha can be gathered after host option syntax is admitted."
);
