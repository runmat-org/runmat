use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const FINDEDGE_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "idx = findedge(G, s, t)",
        &[
            param("G", BuiltinParamType::Any, "Graph or digraph object."),
            param("s", BuiltinParamType::Any, "Graph argument."),
            param("t", BuiltinParamType::Any, "Graph argument."),
        ],
        &[param(
            "idx",
            BuiltinParamType::NumericScalar,
            "Graph result.",
        )],
    )],
    BuiltinOutputMode::Fixed,
);
