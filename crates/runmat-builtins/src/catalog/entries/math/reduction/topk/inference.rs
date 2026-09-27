use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

/// Read exact integer literals without routing wide controls through binary64.
fn control(request: &CallRequest, index: usize) -> Option<usize> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::Bool(value) => Some(usize::from(*value)),
        LiteralValue::Integer { text, .. } => text.parse().ok(),
        literal => {
            let value = LiteralContext::numeric_from_literal(literal)?;
            let rounded = value.round();
            (value.is_finite()
                && value >= 0.0
                && (value - rounded).abs() <= 1e-6
                && rounded < usize::MAX as f64)
                .then_some(rounded as usize)
        }
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        return finish_fixed_outputs(
            entry,
            request,
            vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2],
            vec![argument_error(
                "RM-TOPK-ARITY",
                "input array and k are required",
                0,
            )],
        );
    };
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error("RM-TOPK-ARITY", "k is required", 1));
    }
    if !matches!(
        input.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) || input.storage == StorageFact::Sparse
        || input.numeric().is_some_and(|n| {
            n.class.integer_class().is_some() && n.domain == NumericDomain::Complex
        })
    {
        diagnostics.push(argument_error(
            "RM-TOPK-INPUT",
            "expected dense floating, real integer, or logical input",
            0,
        ));
    }
    let k = control(request, 1);
    if let Some(value) = request.arguments.get(1) {
        if value.shape.element_count().is_some_and(|count| count != 1)
            || matches!(value.kind, ValueKindFact::String | ValueKindFact::Character)
            || request.literals.literal_args.get(1).is_some_and(|literal| {
                !matches!(literal, LiteralValue::Unknown | LiteralValue::Bool(_)) && k.is_none()
            })
        {
            diagnostics.push(argument_error(
                "RM-TOPK-K",
                "k must be a nonnegative integer scalar in range",
                1,
            ));
        }
    }
    let mut dimension = None;
    let mut dimension_present = false;
    let mut unknown_options = false;
    let mut index = 2;
    while index < request.arguments.len() {
        match request.literals.literal_string_at(index).as_deref() {
            Some("comparisonmethod") => {
                if index + 1 == request.arguments.len() {
                    diagnostics.push(argument_error(
                        "RM-TOPK-OPTION",
                        "ComparisonMethod requires a value",
                        index,
                    ));
                } else if let Some(method) = request.literals.literal_string_at(index + 1) {
                    if !["auto", "abs", "magnitude", "real"].contains(&method.as_str()) {
                        diagnostics.push(argument_error(
                            "RM-TOPK-OPTION",
                            "unsupported ComparisonMethod",
                            index + 1,
                        ));
                    }
                }
                index += 2;
            }
            Some(_) => {
                diagnostics.push(argument_error(
                    "RM-TOPK-OPTION",
                    "unrecognized option",
                    index,
                ));
                index += 1;
            }
            None => {
                if dimension_present {
                    // An unknown string may start a name-value option, not a dimension.
                    unknown_options = true;
                } else {
                    dimension = control(request, index);
                    dimension_present = true;
                    if dimension == Some(0)
                        || request.arguments[index]
                            .shape
                            .element_count()
                            .is_some_and(|n| n != 1)
                        || request
                            .literals
                            .literal_args
                            .get(index)
                            .is_some_and(|literal| {
                                !matches!(literal, LiteralValue::Unknown) && dimension.is_none()
                            })
                    {
                        diagnostics.push(argument_error(
                            "RM-TOPK-DIM",
                            "dimension must be a positive integer scalar",
                            index,
                        ));
                    }
                }
                index += 1;
            }
        }
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = input.kind.clone();
    output.layout = LayoutFact::ColumnMajor;
    output.shape = if unknown_options {
        ShapeFact::Unknown
    } else {
        shape(&input.shape, k, dimension, dimension_present)
    };
    output.residency = if input.residency == ResidencyFact::Host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    output.storage = storage(&output);
    let mut indices = output.clone();
    indices.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    indices.storage = storage(&indices);
    finish_fixed_outputs(entry, request, vec![output, indices], diagnostics)
}

fn shape(
    input: &ShapeFact,
    k: Option<usize>,
    dimension: Option<usize>,
    explicit: bool,
) -> ShapeFact {
    let Some(mut dims) = input.known_dims().map(|dims| dims.to_vec()) else {
        return ShapeFact::Unknown;
    };
    while dims.len() < 2 {
        dims.push(Some(1));
    }
    let axis = if explicit {
        let Some(dimension) = dimension.filter(|d| *d > 0) else {
            return ShapeFact::Unknown;
        };
        dimension - 1
    } else {
        let mut chosen = None;
        for (index, dim) in dims.iter().enumerate() {
            match dim {
                Some(n) if *n > 1 => {
                    chosen = Some(index);
                    break;
                }
                None => return ShapeFact::Unknown,
                _ => {}
            }
        }
        chosen.unwrap_or(0)
    };
    if axis < dims.len() {
        dims[axis] = match (k, dims[axis]) {
            (Some(0), _) => Some(0),
            (Some(k), Some(n)) => Some(k.min(n)),
            _ => None,
        };
    } else if k.is_none_or(|k| k == 0) {
        // A zero selection introduces the trailing zero axis; an arbitrarily
        // large runtime dimension must not allocate a compiler shape vector.
        return ShapeFact::Unknown;
    }
    ShapeFact::from(dims)
}

fn storage(value: &ValueFact) -> StorageFact {
    if value.residency != ResidencyFact::Host {
        return StorageFact::Unknown;
    }
    match value.shape.element_count() {
        Some(1) if matches!(value.kind, ValueKindFact::Unknown) => StorageFact::Unknown,
        Some(1)
            if !value
                .numeric()
                .is_some_and(|n| n.class == NumericClass::Single) =>
        {
            StorageFact::Scalar
        }
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    }
}

#[cfg(test)]
mod tests;
