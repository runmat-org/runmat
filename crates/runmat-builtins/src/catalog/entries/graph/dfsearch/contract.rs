use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const DFSEARCH_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[signature(
        "nodes = dfsearch(G, start)",
        &[
            param("G", BuiltinParamType::Any, "Graph or digraph object."),
            param("start", BuiltinParamType::Any, "Graph argument."),
        ],
        &[param("nodes", BuiltinParamType::Any, "Graph result.")],
    )],
    BuiltinOutputMode::Fixed,
);
