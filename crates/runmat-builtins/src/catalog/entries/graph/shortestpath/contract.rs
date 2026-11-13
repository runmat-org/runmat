use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const SHORTESTPATH_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[
        signature(
            "path = shortestpath(G, s, t)",
            &[
                param("G", BuiltinParamType::Any, "Graph or digraph object."),
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
            ],
            &[param("path", BuiltinParamType::Any, "Graph result.")],
        ),
        signature(
            "[path, d] = shortestpath(G, s, t)",
            &[
                param("G", BuiltinParamType::Any, "Graph or digraph object."),
                param("s", BuiltinParamType::Any, "Graph argument."),
                param("t", BuiltinParamType::Any, "Graph argument."),
            ],
            &[
                param("path", BuiltinParamType::Any, "Graph result."),
                param("d", BuiltinParamType::NumericScalar, "Graph result."),
            ],
        ),
    ],
    BuiltinOutputMode::ByRequestedOutputCount,
);
