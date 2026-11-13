//! Static output facts for modal values, frequencies, and tied-value cells.
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DimensionFact, DynamicReason,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Debug)]
enum Axis {
    Default,
    All,
    Dimensions(Vec<usize>),
    Dynamic,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let input = request.arguments.first();
    if input.is_none() {
        diagnostics.push(error("ARITY", "mode requires input data"));
    }
    if request.arguments.len() > 2 {
        diagnostics.push(error(
            "ARITY",
            "mode accepts at most one dimension selector",
        ));
    }
    let axis = parse_axis(request.literals.literal_args.get(1), &mut diagnostics);
    let shape = input
        .map(|fact| reduced_shape(&fact.shape, &axis))
        .unwrap_or(ShapeFact::Unknown);
    let m_kind = input.map_or(ValueKindFact::Unknown, |fact| match fact.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            class,
        }) => {
            if fact.shape.element_count() == Some(0) {
                // Empty reduction produces a NaN double, while identity
                // selection can retain the input class at runtime.
                ValueKindFact::Unknown
            } else {
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                })
            }
        }
        ValueKindFact::Logical => {
            if fact.shape.element_count() == Some(0) {
                ValueKindFact::Unknown
            } else {
                ValueKindFact::Logical
            }
        }
        ValueKindFact::Unknown => ValueKindFact::Unknown,
        _ => {
            diagnostics.push(error("INPUT", "mode requires real numeric or logical data"));
            ValueKindFact::Unknown
        }
    });
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let m_storage = if matches!(
        m_kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        })
    ) && shape.element_count() == Some(1)
    {
        StorageFact::Dense
    } else {
        storage
    };
    let mut m = output(m_kind.clone(), shape.clone(), m_storage);
    let mut f = output(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape.clone(),
        storage,
    );
    let c = output(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Unknown,
    );
    if let Some(input) = input {
        m.residency = output_residency(&input.residency);
        f.residency = output_residency(&input.residency);
    }
    let mut contract = CallContract::fixed(vec![m, f, c]);
    // The current runtime pads requests after the three defined outputs with
    // double zero scalars; this is a compatibility behavior, not a fourth
    // semantic output.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn output(kind: ValueKindFact, shape: ShapeFact, storage: StorageFact) -> ValueFact {
    if matches!(kind, ValueKindFact::Unknown) {
        let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
        fact.shape = shape;
        fact.storage = storage;
        fact
    } else {
        ValueFact::proven(kind, shape, storage)
    }
}

fn output_residency(input: &ResidencyFact) -> ResidencyFact {
    match input {
        ResidencyFact::Host => ResidencyFact::Host,
        ResidencyFact::Device { .. } | ResidencyFact::Unknown | ResidencyFact::Remote { .. } => {
            ResidencyFact::Unknown
        }
    }
}

fn parse_axis(literal: Option<&LiteralValue>, diagnostics: &mut Vec<InferenceDiagnostic>) -> Axis {
    match literal {
        None | Some(LiteralValue::Empty) => Axis::Default,
        Some(
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text),
        ) => {
            if text.trim().eq_ignore_ascii_case("all") {
                Axis::All
            } else {
                diagnostics.push(error("DIMENSION", "unsupported dimension option"));
                Axis::Dynamic
            }
        }
        Some(LiteralValue::Vector(values)) => {
            let mut dims = Vec::new();
            for value in values {
                let Some(dim) = literal_dimension(value) else {
                    return Axis::Dynamic;
                };
                if dim == 0 {
                    diagnostics.push(error("DIMENSION", "dimensions must be positive integers"));
                    return Axis::Dynamic;
                }
                dims.push(dim);
            }
            if dims.is_empty() {
                Axis::Default
            } else {
                Axis::Dimensions(dims)
            }
        }
        Some(
            value @ (LiteralValue::Number(_)
            | LiteralValue::Real { .. }
            | LiteralValue::Integer { .. }
            | LiteralValue::Bool(_)),
        ) => match literal_dimension(value) {
            Some(dim) if dim > 0 => Axis::Dimensions(vec![dim]),
            _ => {
                diagnostics.push(error("DIMENSION", "dimension must be a positive integer"));
                Axis::Dynamic
            }
        },
        Some(LiteralValue::Unknown) => Axis::Dynamic,
        Some(_) => {
            diagnostics.push(error("DIMENSION", "unsupported dimension selector"));
            Axis::Dynamic
        }
    }
}

fn literal_dimension(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Bool(value) => Some(if *value { 1 } else { 0 }),
        LiteralValue::Integer { text, .. } => text.parse::<usize>().ok(),
        other => LiteralContext::numeric_from_literal(other).and_then(|value| {
            (value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < usize::MAX as f64)
                .then_some(value as usize)
        }),
    }
}

fn reduced_shape(input: &ShapeFact, axis: &Axis) -> ShapeFact {
    match axis {
        Axis::All => return ShapeFact::Scalar,
        Axis::Dynamic => return ShapeFact::Unknown,
        Axis::Default | Axis::Dimensions(_) => {}
    }
    let ShapeFact::Shaped { dims } = input else {
        return match input {
            ShapeFact::Scalar => ShapeFact::Scalar,
            ShapeFact::Ranked { rank } => ShapeFact::Ranked { rank: *rank },
            ShapeFact::Unknown | ShapeFact::Shaped { .. } => ShapeFact::Unknown,
        };
    };
    if matches!(axis, Axis::Default)
        && dims.as_slice() == [DimensionFact::Known(0), DimensionFact::Known(0)]
    {
        return ShapeFact::Scalar;
    }
    let mut out = dims.clone();
    let axes = match axis {
        Axis::Dimensions(values) => values.clone(),
        Axis::Default => {
            let mut selected = 1;
            for (index, dim) in dims.iter().enumerate() {
                match dim {
                    DimensionFact::Known(1) => {}
                    DimensionFact::Known(_) => {
                        selected = index + 1;
                        break;
                    }
                    DimensionFact::Symbolic(_) | DimensionFact::Unknown => {
                        return ShapeFact::Ranked { rank: dims.len() };
                    }
                }
            }
            vec![selected]
        }
        Axis::All | Axis::Dynamic => unreachable!("handled above"),
    };
    for selected in axes {
        if let Some(dim) = selected.checked_sub(1).and_then(|index| out.get_mut(index)) {
            *dim = DimensionFact::Known(1);
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
    use runmat_types::{OutputSelection, RequestedOutputCount};

    #[test]
    fn ranked_shapes_are_not_expanded_for_remote_dimensions() {
        let huge = ShapeFact::Ranked { rank: usize::MAX };
        assert_eq!(reduced_shape(&huge, &Axis::Default), huge);
        assert_eq!(
            reduced_shape(&huge, &Axis::Dimensions(vec![usize::MAX])),
            huge
        );
        assert_eq!(reduced_shape(&huge, &Axis::All), ShapeFact::Scalar);
    }

    #[test]
    fn exact_integer_mode_and_double_frequency_have_distinct_facts() {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(4), Some(3)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let result = infer(&request, &super::super::MODE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs.len(), 3);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(
            result.outputs[1].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        );
        assert!(matches!(result.outputs[2].kind, ValueKindFact::Cell(_)));
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(3)])
        );
    }
}
