use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const INDEGREE_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "d = indegree(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param("d", BuiltinParamType::NumericScalar, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
