use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const OUTDEGREE_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "d = outdegree(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param(
            "d",
            BuiltinParamType::NumericArray,
            "One double out-degree per node, in an n-by-1 column vector.",
        )],
    )],
    BuiltinOutputMode::Fixed,
);
