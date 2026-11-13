use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const DIGRAPH_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[
        signature(
            "G = digraph()",
            &[],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = digraph(s, t, weights)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
                param("weights", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = digraph(s, t, names)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
                param("names", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = digraph(s, t, weights, names)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
                param("weights", BuiltinParamType::Any, "Graph argument."),
                param("names", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = digraph(A, names)",
            &[
                param("A", BuiltinParamType::Any, "Graph argument."),
                param("names", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = digraph(s, t)",
            &[
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "G = digraph(A)",
            &[param("A", BuiltinParamType::Any, "Graph argument.")],
            &[param("G", BuiltinParamType::Any, "Graph result.")],
        ),
    ],
    BuiltinOutputMode::Fixed,
);
