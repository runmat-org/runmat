use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const PREDECESSORS_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "P = predecessors(G, node)",
        &[
            param("G", BuiltinParamType::Any, "Graph or digraph object."),
            param("node", BuiltinParamType::Any, "Graph argument."),
        ],
        &[param("P", BuiltinParamType::Any, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
