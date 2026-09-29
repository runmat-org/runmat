use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const TOPOSORT_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "order = toposort(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Directed acyclic digraph object.",
        )],
        &[param("order", BuiltinParamType::Any, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
