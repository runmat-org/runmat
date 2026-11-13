use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const NEIGHBORS_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "N = neighbors(G, node)",
        &[
            param("G", BuiltinParamType::Any, "Graph or digraph object."),
            param("node", BuiltinParamType::Any, "Graph argument."),
        ],
        &[param("N", BuiltinParamType::Any, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
