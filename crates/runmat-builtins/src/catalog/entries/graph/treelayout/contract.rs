use super::super::shared::{descriptor, param, signature};
use crate::*;

pub const TREELAYOUT_DESCRIPTOR: BuiltinDescriptor = descriptor(
    &[
        signature(
            "x = treelayout(parent)",
            &[param(
                "parent",
                BuiltinParamType::NumericArray,
                "Zero-rooted one-based parent vector.",
            )],
            &[param("x", BuiltinParamType::NumericArray, "Graph result.")],
        ),
        signature(
            "[x, y] = treelayout(parent)",
            &[param(
                "parent",
                BuiltinParamType::NumericArray,
                "Zero-rooted one-based parent vector.",
            )],
            &[
                param("x", BuiltinParamType::NumericArray, "Graph result."),
                param("y", BuiltinParamType::NumericArray, "Graph result."),
            ],
        ),
    ],
    BuiltinOutputMode::ByRequestedOutputCount,
);
