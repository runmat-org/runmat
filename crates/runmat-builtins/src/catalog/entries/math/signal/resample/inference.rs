use super::super::sample_rate_facts::structural_integer_literal;
use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESAMPLE-ARITY",
            "resample expects X, P, Q, and optional filter controls",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let input = request.arguments.first();
    if let Some(value) = input {
        let supported = matches!(
            &value.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) && !matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque);
        if !supported {
            diagnostics.push(argument_error(
                "RM-CATALOG-RESAMPLE-INPUT",
                "resample requires a dense single, double, or logical input",
                0,
            ));
        }
    }
    let p = positive(request, 1, "P", &mut diagnostics);
    let q = positive(request, 2, "Q", &mut diagnostics);
    let dimension = explicit_dimension(request, &mut diagnostics);
    let output = match input {
        Some(input) if diagnostics.is_empty() => output(input, p, q, dimension),
        Some(_) => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut filter = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
        },
        StorageFact::Dense,
    );
    filter.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output, filter]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn positive(
    request: &CallRequest,
    index: usize,
    name: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    let value = request.arguments.get(index)?;
    if value.shape.is_definitely_nonscalar()
        || !matches!(
            &value.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        )
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESAMPLE-RATE",
            format!("{name} must be a positive integer scalar"),
            index,
        ));
        return None;
    }
    let literal = request.literals.literal_args.get(index)?;
    let parsed = structural_integer_literal(literal, true)?;
    match parsed {
        Ok(parsed) => Some(parsed),
        Err(()) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-RESAMPLE-RATE",
                format!("{name} must be a positive platform integer"),
                index,
            ));
            None
        }
    }
}

#[derive(Clone, Copy)]
enum DimensionSelection {
    Default,
    Known(usize),
    Dynamic,
}

fn explicit_dimension(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> DimensionSelection {
    let mut dynamic = false;
    for index in 3..request.arguments.len() {
        let text = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text);
        if text
            .as_deref()
            .is_some_and(|text| text.eq_ignore_ascii_case("dimension"))
        {
            let value_index = index + 1;
            let Some(value) = request.arguments.get(value_index) else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RESAMPLE-DIMENSION",
                    "Dimension requires a value",
                    index,
                ));
                return DimensionSelection::Dynamic;
            };
            if value.shape.is_definitely_nonscalar() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RESAMPLE-DIMENSION",
                    "Dimension must be a positive integer scalar",
                    value_index,
                ));
                return DimensionSelection::Dynamic;
            }
            if !matches!(
                &value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Unknown
            ) || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
                || matches!(
                    value.residency,
                    ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
                )
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RESAMPLE-DIMENSION",
                    "Dimension must be a host real numeric scalar",
                    value_index,
                ));
                return DimensionSelection::Dynamic;
            }
            let Some(literal) = request.literals.literal_args.get(value_index) else {
                return DimensionSelection::Dynamic;
            };
            let Some(parsed) = structural_integer_literal(literal, true) else {
                return DimensionSelection::Dynamic;
            };
            let Ok(parsed) = parsed else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RESAMPLE-DIMENSION",
                    "Dimension must be a positive platform integer",
                    value_index,
                ));
                return DimensionSelection::Dynamic;
            };
            return DimensionSelection::Known(parsed - 1);
        }
        if text.is_some() {
            diagnostics.push(argument_error(
                "RM-CATALOG-RESAMPLE-OPTION",
                "Unknown resample option name",
                index,
            ));
            return DimensionSelection::Dynamic;
        }
        if request
            .literals
            .literal_args
            .get(index)
            .is_none_or(|value| matches!(value, &LiteralValue::Unknown))
        {
            dynamic = true;
        }
    }
    if dynamic {
        DimensionSelection::Dynamic
    } else {
        DimensionSelection::Default
    }
}

fn output(
    input: &ValueFact,
    p: Option<usize>,
    q: Option<usize>,
    dimension: DimensionSelection,
) -> ValueFact {
    let fact = match &input.kind {
        ValueKindFact::Numeric(fact) => *fact,
        ValueKindFact::Logical => NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let shape = if matches!(dimension, DimensionSelection::Dynamic) {
        ShapeFact::Unknown
    } else {
        match &input.shape {
            ShapeFact::Scalar | ShapeFact::Shaped { .. } => {
                let mut dims = match &input.shape {
                    ShapeFact::Shaped { dims } => dims.clone(),
                    _ => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
                };
                if dims.is_empty() {
                    dims = vec![DimensionFact::Known(1), DimensionFact::Known(1)];
                }
                let candidates = match dimension {
                    DimensionSelection::Known(axis) if axis < dims.len() => vec![axis],
                    DimensionSelection::Known(_) => {
                        let mut output = ValueFact::proven(
                            ValueKindFact::Numeric(fact),
                            ShapeFact::Unknown,
                            StorageFact::Unknown,
                        );
                        output.residency = if input.residency == ResidencyFact::Host {
                            ResidencyFact::Host
                        } else {
                            ResidencyFact::Unknown
                        };
                        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
                        return output;
                    }
                    DimensionSelection::Default => {
                        let mut axes = Vec::new();
                        let mut definite = false;
                        for (axis, dim) in dims.iter().enumerate() {
                            match dim {
                                DimensionFact::Known(1) => {}
                                DimensionFact::Known(_) => {
                                    axes.push(axis);
                                    definite = true;
                                    break;
                                }
                                DimensionFact::Symbolic(_) | DimensionFact::Unknown => {
                                    axes.push(axis)
                                }
                            }
                        }
                        if !definite {
                            axes.push(0);
                        }
                        axes.sort_unstable();
                        axes.dedup();
                        axes
                    }
                    DimensionSelection::Dynamic => unreachable!(),
                };
                let unique_axis = candidates.len() == 1;
                for axis in candidates {
                    let transformed = match (dims[axis].clone(), p, q) {
                        (DimensionFact::Known(n), Some(p), Some(q)) => n
                            .checked_mul(p)
                            .and_then(|v| v.checked_add(q - 1))
                            .map(|v| DimensionFact::Known(v / q))
                            .unwrap_or(DimensionFact::Unknown),
                        _ => DimensionFact::Unknown,
                    };
                    if unique_axis {
                        dims[axis] = transformed;
                    } else if dims[axis] != transformed {
                        dims[axis] = DimensionFact::Unknown;
                    }
                }
                ShapeFact::Shaped { dims }
            }
            _ => ShapeFact::Unknown,
        }
    };
    let storage = match shape.element_count() {
        Some(1) if fact.class == NumericClass::Double && input.residency == ResidencyFact::Host => {
            StorageFact::Scalar
        }
        Some(1) if input.residency != ResidencyFact::Host => StorageFact::Unknown,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(ValueKindFact::Numeric(fact), shape, storage);
    if matches!(&output.shape, ShapeFact::Unknown)
        || matches!(&output.shape, ShapeFact::Shaped { dims } if dims.iter().any(|dim| matches!(dim, DimensionFact::Unknown)))
    {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output.residency = if input.residency == ResidencyFact::Host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    output
}
