use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const ADJACENCY_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "A = adjacency(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param("A", BuiltinParamType::NumericArray, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
