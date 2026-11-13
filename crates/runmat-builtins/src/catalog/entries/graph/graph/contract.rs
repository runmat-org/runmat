use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const GRAPH_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[
        signature(
            "G = graph()",
            &[],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = graph(s, t, weights)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
                param("weights", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = graph(s, t, names)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
                param("names", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = graph(A, names)",
            &[
                param("A", BuiltinParamType::Any, "Graph argument."),
                param("names", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = graph(A, triangle)",
            &[
                param("A", BuiltinParamType::Any, "Graph argument."),
                param("triangle", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = graph(s, t)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = graph(s, t, weights, names)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
                param("weights", BuiltinParamType::Any, "Graph argument."),
                param("names", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = graph(A)",
            &[param("A", BuiltinParamType::Any, "Graph argument.")],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
    ],
    BuiltinOutputMode::Fixed,
);
