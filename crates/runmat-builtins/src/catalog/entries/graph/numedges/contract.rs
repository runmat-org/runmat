use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const NUMEDGES_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "m = numedges(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param("m", BuiltinParamType::NumericScalar, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
