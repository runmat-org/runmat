use super::super::graph::shared::{descriptor, param, signature};
use crate::*;

pub const TOPOSORT_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "order = toposort(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param("order", BuiltinParamType::Any, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
