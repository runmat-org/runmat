use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const SUCCESSORS_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "S = successors(G, node)",
        &[
            param("G", BuiltinParamType::Any, "Graph or digraph object."),
            param("node", BuiltinParamType::Any, "Graph argument."),
        ],
        &[param("S", BuiltinParamType::Any, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
