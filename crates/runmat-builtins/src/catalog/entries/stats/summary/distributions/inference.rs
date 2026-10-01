//! Static facts for scalar-broadcast distribution operations.

use super::policy::{DistributionKind, DistributionTail, InverseDistributionFamily};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain, NumericFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: DistributionKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let arguments = &request.arguments;
    let numeric_start = usize::from(kind == DistributionKind::GenericInv);
    let mut numeric_end = arguments.len();

    if kind == DistributionKind::GenericInv {
        validate_family(request, &mut diagnostics);
    } else if kind.supports_upper_tail() && numeric_end > 0 {
        let last = numeric_end - 1;
        match request.literals.literal_args.get(last) {
            Some(
                LiteralValue::Character(text)
                | LiteralValue::String(text)
                | LiteralValue::Keyword(text),
            ) if DistributionTail::from_keyword(text) == Some(DistributionTail::Upper) => {
                numeric_end -= 1;
            }
            _ if matches!(
                &arguments[last].kind,
                ValueKindFact::Character | ValueKindFact::String
            ) =>
            {
                numeric_end -= 1;
                if request
                    .literals
                    .literal_args
                    .get(last)
                    .is_some_and(|literal| !matches!(literal, LiteralValue::Unknown))
                {
                    diagnostics.push(error(
                        "TAIL",
                        "the only supported tail selector is upper",
                        last,
                    ));
                }
            }
            _ => {}
        }
    }

    if kind != DistributionKind::GenericInv && !kind.accepts_numeric_count(numeric_end) {
        diagnostics.push(error(
            "ARITY",
            "unsupported number of distribution arguments",
            numeric_end.saturating_sub(1),
        ));
    }
    if numeric_end < numeric_start {
        diagnostics.push(error("ARITY", "a numerical input is required", 0));
    }

    let numeric = arguments.get(numeric_start..numeric_end).unwrap_or(&[]);
    let mut shape = ShapeFact::Scalar;
    let mut any_single = false;
    let mut unknown_class = false;
    let mut unknown_input = kind == DistributionKind::GenericInv
        && match arguments.first().map(|argument| &argument.kind) {
            Some(ValueKindFact::Object(object)) => object.runtime_class.is_none(),
            Some(ValueKindFact::String | ValueKindFact::Character) => {
                request.literals.literal_string_at(0).is_none()
            }
            _ => true,
        };
    for (offset, argument) in numeric.iter().enumerate() {
        match &argument.kind {
            ValueKindFact::Numeric(fact) if fact.domain == NumericDomain::Real => {
                any_single |= fact.class == NumericClass::Single;
            }
            ValueKindFact::Logical => {}
            ValueKindFact::Unknown => {
                unknown_class = true;
                unknown_input = true;
            }
            _ => diagnostics.push(error(
                "CLASS",
                "distribution numerical inputs must be real numeric or logical values",
                numeric_start + offset,
            )),
        }
        if kind != DistributionKind::ChiSquareCdf {
            shape = broadcast(&shape, &argument.shape, &mut diagnostics);
        }
    }
    if kind == DistributionKind::ChiSquareCdf {
        shape = chi_square_shape(arguments, &mut diagnostics);
    }

    let class = if any_single {
        Some(NumericClass::Single)
    } else if unknown_class {
        None
    } else {
        Some(NumericClass::Double)
    };
    let mut output = match class {
        Some(class) if !unknown_input && !matches!(&shape, ShapeFact::Unknown) => {
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                shape,
                StorageFact::Unknown,
            )
        }
        _ => {
            let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
            if let Some(class) = class {
                result.kind = ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                });
            }
            result.shape = shape;
            result
        }
    };
    let host_inputs = numeric
        .iter()
        .all(|argument| argument.residency == ResidencyFact::Host);
    output.storage = match (class, output.shape.element_count()) {
        (Some(NumericClass::Double), Some(1)) if host_inputs => StorageFact::Scalar,
        (Some(NumericClass::Double), Some(1)) => StorageFact::Unknown,
        (Some(_), Some(_)) => StorageFact::Dense,
        _ => StorageFact::Unknown,
    };
    output.residency = ResidencyFact::Unknown;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    inference.diagnostics.extend(diagnostics);
    inference
}

fn validate_family(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let arguments = &request.arguments;
    if arguments.len() < 2 {
        diagnostics.push(error(
            "ARITY",
            "icdf requires a family or fitted distribution and a probability",
            0,
        ));
        return;
    }
    match &arguments[0].kind {
        ValueKindFact::Object(object) => {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| !class.is(runmat_types::standard::PROBABILITY_DISTRIBUTION))
            {
                diagnostics.push(error(
                    "FAMILY",
                    "icdf requires a fitted probability distribution object",
                    0,
                ));
            }
            if arguments.len() != 2 {
                diagnostics.push(error(
                    "ARITY",
                    "the fitted-distribution form accepts exactly two inputs",
                    arguments.len() - 1,
                ));
            }
        }
        ValueKindFact::String | ValueKindFact::Character => {
            if let Some(name) = request.literals.literal_string_at(0) {
                match InverseDistributionFamily::from_keyword(&name) {
                    Some(family) if !family.accepts_parameter_count(arguments.len() - 2) => {
                        diagnostics.push(error(
                            "ARITY",
                            "incorrect parameter count for this distribution",
                            arguments.len() - 1,
                        ))
                    }
                    None => diagnostics.push(error("FAMILY", "unsupported distribution family", 0)),
                    _ => {}
                }
            }
            if arguments.len() > 4 {
                diagnostics.push(error(
                    "ARITY",
                    "too many distribution parameters",
                    arguments.len() - 1,
                ));
            }
        }
        ValueKindFact::Unknown => {}
        _ => diagnostics.push(error(
            "FAMILY",
            "icdf requires a distribution name or fitted object",
            0,
        )),
    }
}

fn broadcast(
    left: &ShapeFact,
    right: &ShapeFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let (Some(left_rank), Some(right_rank)) = (left.rank(), right.rank()) else {
        return ShapeFact::Unknown;
    };
    let rank = left_rank.max(right_rank).max(2);
    if matches!(left, ShapeFact::Ranked { .. }) || matches!(right, ShapeFact::Ranked { .. }) {
        return ShapeFact::Ranked { rank };
    }
    let mut dims = Vec::new();
    if dims.try_reserve_exact(rank).is_err() {
        return ShapeFact::Ranked { rank };
    }
    for axis in 0..rank {
        let a = dimension(left, axis, rank);
        let b = dimension(right, axis, rank);
        dims.push(match (&a, &b) {
            (DimensionFact::Known(a), DimensionFact::Known(b)) if a == b => {
                DimensionFact::Known(*a)
            }
            (DimensionFact::Known(1), other) | (other, DimensionFact::Known(1)) => other.clone(),
            (DimensionFact::Known(_), DimensionFact::Known(_)) => {
                diagnostics.push(error(
                    "BROADCAST",
                    "distribution inputs have incompatible sizes",
                    axis,
                ));
                DimensionFact::Unknown
            }
            _ if a == b => a,
            _ => DimensionFact::Unknown,
        });
    }
    ShapeFact::Shaped { dims }
}

fn chi_square_shape(
    arguments: &[ValueFact],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    let (Some(left), Some(right)) = (arguments.first(), arguments.get(1)) else {
        return ShapeFact::Unknown;
    };
    if left.is_scalar() {
        return right.shape.clone();
    }
    if right.is_scalar() {
        return left.shape.clone();
    }
    if matches!(&left.shape, ShapeFact::Unknown | ShapeFact::Ranked { .. })
        || matches!(&right.shape, ShapeFact::Unknown | ShapeFact::Ranked { .. })
    {
        return ShapeFact::Unknown;
    }
    if left.shape.is_proven_equivalent(&right.shape) {
        left.shape.clone()
    } else {
        let known = |shape: &ShapeFact| matches!(shape, ShapeFact::Shaped { dims } if dims.iter().all(|dim| matches!(dim, DimensionFact::Known(_))));
        if known(&left.shape) && known(&right.shape) {
            diagnostics.push(error(
                "BROADCAST",
                "chi2cdf inputs must have the same size unless one is scalar",
                1,
            ));
        }
        ShapeFact::Unknown
    }
}

fn dimension(shape: &ShapeFact, axis: usize, rank: usize) -> DimensionFact {
    match shape {
        ShapeFact::Scalar => DimensionFact::Known(1),
        ShapeFact::Shaped { dims } if rank >= 2 && dims.len() == 1 => {
            if axis == 1 {
                dims[0].clone()
            } else {
                DimensionFact::Known(1)
            }
        }
        ShapeFact::Shaped { dims } => dims.get(axis).cloned().unwrap_or(DimensionFact::Known(1)),
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => DimensionFact::Unknown,
    }
}

fn error(code: &'static str, message: &'static str, argument: usize) -> InferenceDiagnostic {
    let mut diagnostic = InferenceDiagnostic::error(code, message);
    diagnostic.argument = Some(argument);
    diagnostic
}
