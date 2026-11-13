use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const CONNCOMP_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "bins = conncomp(G)",
        &[param(
            "G",
            BuiltinParamType::Any,
            "Graph or digraph object.",
        )],
        &[param(
            "bins",
            BuiltinParamType::NumericArray,
            "Graph result.",
        )],
    )],
    BuiltinOutputMode::Fixed,
);
