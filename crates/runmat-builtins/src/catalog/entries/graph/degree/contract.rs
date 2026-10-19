use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const DEGREE_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "d = degree(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param(
            "d",
            BuiltinParamType::NumericArray,
            "One double degree per node, in an n-by-1 column vector.",
        )],
    )],
    BuiltinOutputMode::Fixed,
);
