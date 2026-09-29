use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const DISTANCES_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[
        signature(
            "D = distances(G, s)",
            &[
                param("G", BuiltinParamType::Any, "Graph or digraph object."),
                param("s", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("D", BuiltinParamType::NumericArray, "Graph result.")],
        ),
        signature(
            "D = distances(G)",
            &[param(
                "G",
                BuiltinParamType::Any,
                "Graph or digraph object.",
            )],
            &[param("D", BuiltinParamType::NumericArray, "Graph result.")],
        ),
        signature(
            "D = distances(G, s, t)",
            &[
                param("G", BuiltinParamType::Any, "Graph or digraph object."),
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("D", BuiltinParamType::NumericArray, "Graph result.")],
        ),
    ],
    BuiltinOutputMode::Fixed,
);
