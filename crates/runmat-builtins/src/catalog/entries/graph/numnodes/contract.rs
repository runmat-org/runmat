use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const NUMNODES_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "n = numnodes(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param("n", BuiltinParamType::NumericScalar, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
