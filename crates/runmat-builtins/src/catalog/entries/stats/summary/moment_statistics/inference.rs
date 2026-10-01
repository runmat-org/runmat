//! Static facts for the flagged deviation and moment reductions.
use super::policy::MomentStatisticKind;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

#[derive(Clone, Debug)]
enum Selector {
    Default,
    All,
    Dimensions(Vec<usize>),
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingMode {
    Include,
    Omit,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: MomentStatisticKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let input = request.arguments.first();
    if input.is_none() {
        diagnostics.push(error("ARITY", "input data is required"));
    }
    let class = input.and_then(|fact| match fact.kind {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }) => Some(match class {
            NumericClass::Single => NumericClass::Single,
            NumericClass::Double
            | NumericClass::Int8
            | NumericClass::UInt8
            | NumericClass::Int16
            | NumericClass::UInt16
            | NumericClass::Int32
            | NumericClass::UInt32
            | NumericClass::Int64
            | NumericClass::UInt64 => NumericClass::Double,
        }),
        ValueKindFact::Numeric(_) => {
            diagnostics.push(error("COMPLEX", "real numeric input is required"));
            None
        }
        ValueKindFact::Logical => Some(NumericClass::Double),
        ValueKindFact::Unknown => None,
        _ => {
            diagnostics.push(error("INPUT", "numeric input is required"));
            None
        }
    });
    let (selector, _flag, _missing) = parse_options(request, kind, &mut diagnostics);
    let shape = input
        .map(|fact| reduced_shape(&fact.shape, &selector))
        .unwrap_or(ShapeFact::Unknown);
    let mut output = match class {
        Some(class) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Unknown,
        ),
        None => {
            let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.shape = shape;
            output
        }
    };
    output.storage = match output.shape.element_count() {
        Some(1) if class == Some(NumericClass::Single) => StorageFact::Dense,
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn parse_options(
    request: &CallRequest,
    kind: MomentStatisticKind,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (Selector, u8, MissingMode) {
    let mut flag = kind.default_flag();
    let mut selector = Selector::Default;
    let mut missing = MissingMode::Omit;
    let mut index = 1;
    let mut ambiguous_first = false;
    if let Some(first) = request.literals.literal_args.get(1) {
        match first {
            LiteralValue::Empty => index += 1,
            LiteralValue::Bool(value) => {
                flag = u8::from(*value);
                index += 1;
            }
            LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. } => {
                match literal_dimension(first) {
                    Some(value @ 0..=1) => flag = value as u8,
                    _ => diagnostics.push(error("FLAG", "flag must be 0 or 1")),
                }
                index += 1;
            }
            LiteralValue::Unknown => {
                // The first runtime value may be a flag or a dimension.
                // Later source-known options cannot resolve that ambiguity.
                ambiguous_first = true;
                index += 1;
            }
            _ => {}
        }
    }
    let mut saw_selector = false;
    let mut saw_missing = false;
    while index < request.arguments.len() {
        let literal = request
            .literals
            .literal_args
            .get(index)
            .unwrap_or(&LiteralValue::Unknown);
        match literal {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => match text.to_ascii_lowercase().as_str() {
                "all" if !saw_selector => {
                    selector = Selector::All;
                    saw_selector = true;
                }
                "omitnan" | "omitmissing" if !saw_missing => {
                    missing = MissingMode::Omit;
                    saw_missing = true;
                }
                "includenan" | "includemissing" if !saw_missing => {
                    missing = MissingMode::Include;
                    saw_missing = true;
                }
                _ => diagnostics.push(error("OPTIONS", "unsupported or duplicate option")),
            },
            LiteralValue::Vector(values) if values.len() == 1 => {
                match literal_dimension(&values[0]) {
                    Some(value @ 0..=1) => flag = value as u8,
                    Some(_) => diagnostics.push(error("FLAG", "flag must be 0 or 1")),
                    None => ambiguous_first = true,
                }
                index += 1;
            }
            LiteralValue::Matrix(rows) if rows.len() == 1 && rows[0].len() == 1 => {
                match literal_dimension(&rows[0][0]) {
                    Some(value @ 0..=1) => flag = value as u8,
                    Some(_) => diagnostics.push(error("FLAG", "flag must be 0 or 1")),
                    None => ambiguous_first = true,
                }
                index += 1;
            }
            LiteralValue::Number(_)
            | LiteralValue::Real { .. }
            | LiteralValue::Integer { .. }
            | LiteralValue::Vector(_) => {
                if saw_selector {
                    diagnostics.push(error("DIMENSION", "only one dimension selector is allowed"));
                } else {
                    selector = parse_selector(literal, diagnostics);
                    saw_selector = true;
                }
            }
            LiteralValue::Unknown => selector = Selector::Dynamic,
            _ => diagnostics.push(error("OPTIONS", "unsupported reduction option")),
        }
        index += 1;
    }
    if ambiguous_first {
        selector = Selector::Dynamic;
    }
    (selector, flag, missing)
}

fn parse_selector(literal: &LiteralValue, diagnostics: &mut Vec<InferenceDiagnostic>) -> Selector {
    let values = match literal {
        LiteralValue::Vector(values) => values.as_slice(),
        other => std::slice::from_ref(other),
    };
    let mut dims = Vec::new();
    for value in values {
        if matches!(value, LiteralValue::Unknown) {
            return Selector::Dynamic;
        }
        match literal_dimension(value) {
            Some(dim) if dim > 0 => dims.push(dim),
            _ => diagnostics.push(error(
                "DIMENSION",
                "dimensions must be distinct positive integers",
            )),
        }
    }
    if dims.is_empty() {
        diagnostics.push(error("DIMENSION", "at least one dimension is required"));
    }
    Selector::Dimensions(dims)
}

fn literal_dimension(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok(),
        other => LiteralContext::numeric_from_literal(other).and_then(|value| {
            (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < usize::MAX as f64)
                .then_some(value as usize)
        }),
    }
}

fn reduced_shape(input: &ShapeFact, selector: &Selector) -> ShapeFact {
    if matches!(selector, Selector::All) {
        return ShapeFact::Scalar;
    }
    if matches!(selector, Selector::Dynamic) {
        return ShapeFact::Unknown;
    }
    let ShapeFact::Shaped { dims } = input else {
        return match input {
            ShapeFact::Scalar => ShapeFact::Scalar,
            ShapeFact::Ranked { rank } => ShapeFact::Ranked { rank: *rank },
            ShapeFact::Unknown | ShapeFact::Shaped { .. } => ShapeFact::Unknown,
        };
    };
    let mut out = dims.clone();
    let axes = match selector {
        Selector::Default => {
            let mut first = 0;
            for (index, extent) in dims.iter().enumerate() {
                match extent {
                    runmat_types::DimensionFact::Known(1) => {}
                    runmat_types::DimensionFact::Known(_) => {
                        first = index;
                        break;
                    }
                    runmat_types::DimensionFact::Unknown
                    | runmat_types::DimensionFact::Symbolic(_) => {
                        return ShapeFact::Ranked { rank: dims.len() };
                    }
                }
            }
            vec![first]
        }
        Selector::Dimensions(values) => {
            values.iter().filter_map(|dim| dim.checked_sub(1)).collect()
        }
        Selector::All | Selector::Dynamic => unreachable!("handled above"),
    };
    for axis in axes {
        if let Some(dim) = out.get_mut(axis) {
            *dim = runmat_types::DimensionFact::Known(1);
        }
    }
    ShapeFact::Shaped { dims: out }
}

fn error(code: &'static str, message: impl Into<String>) -> InferenceDiagnostic {
    InferenceDiagnostic::error(code, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DimensionFact, OutputSelection, RequestedOutputCount};

    fn options(literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue); literals.len()],
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    #[test]
    fn canonical_flag_defaults_and_static_option_errors() {
        for (kind, expected) in [
            (MomentStatisticKind::MeanAbsoluteDeviation, 0),
            (MomentStatisticKind::Skewness, 1),
            (MomentStatisticKind::Kurtosis, 1),
        ] {
            let request = options(vec![LiteralValue::Unknown]);
            let mut diagnostics = Vec::new();
            let (_, flag, missing) = parse_options(&request, kind, &mut diagnostics);
            assert_eq!(flag, expected);
            assert_eq!(missing, MissingMode::Omit);
            assert!(diagnostics.is_empty());

            let request = options(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]);
            let mut diagnostics = Vec::new();
            parse_options(&request, kind, &mut diagnostics);
            assert!(!diagnostics.is_empty(), "{kind:?} accepted invalid flag");
        }
    }

    #[test]
    fn ambiguity_and_large_rank_remain_symbolic() {
        let huge = ShapeFact::Ranked { rank: usize::MAX };
        assert_eq!(reduced_shape(&huge, &Selector::Default), huge);
        assert_eq!(
            reduced_shape(&huge, &Selector::Dimensions(vec![usize::MAX])),
            huge
        );
        assert_eq!(reduced_shape(&huge, &Selector::All), ShapeFact::Scalar);

        let request = options(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
        ]);
        let mut diagnostics = Vec::new();
        let (selector, _, _) =
            parse_options(&request, MomentStatisticKind::Skewness, &mut diagnostics);
        assert!(matches!(selector, Selector::Dynamic));
        assert_eq!(reduced_shape(&huge, &selector), ShapeFact::Unknown);
    }

    #[test]
    fn first_scalar_array_is_a_flag_not_a_dimension() {
        let request = options(vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(1.0)]),
        ]);
        let mut diagnostics = Vec::new();
        let (selector, flag, _) =
            parse_options(&request, MomentStatisticKind::Skewness, &mut diagnostics);
        assert_eq!(flag, 1);
        assert!(matches!(selector, Selector::Default));
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn concrete_reduction_preserves_unselected_nd_extents() {
        let input = ShapeFact::Shaped {
            dims: vec![
                DimensionFact::Known(2),
                DimensionFact::Known(3),
                DimensionFact::Known(4),
            ],
        };
        assert_eq!(
            reduced_shape(&input, &Selector::Dimensions(vec![2, usize::MAX])),
            ShapeFact::Shaped {
                dims: vec![
                    DimensionFact::Known(2),
                    DimensionFact::Known(1),
                    DimensionFact::Known(4),
                ],
            }
        );
    }

    #[test]
    fn single_element_single_remains_a_typed_tensor() {
        let request = CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(3), Some(1)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let result = infer(
            &request,
            &super::super::MAD_CATALOG_ENTRY,
            MomentStatisticKind::MeanAbsoluteDeviation,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].storage, StorageFact::Dense);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
    }
}
