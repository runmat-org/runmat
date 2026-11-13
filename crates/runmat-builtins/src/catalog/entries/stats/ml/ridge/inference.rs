use super::RidgeScaling;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn invalid_known_parameter(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Vector(values) => values.iter().any(invalid_known_parameter),
        LiteralValue::Integer { text, .. } => text.parse::<i128>().is_ok_and(|value| value < 0),
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            LiteralContext::numeric_from_literal(literal)
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        }
        _ => false,
    }
}

fn vector_length(
    shape: &ShapeFact,
    argument: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            let lengths = dims
                .iter()
                .map(|dim| match dim {
                    DimensionFact::Known(n) => Some(*n),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if lengths[0].is_some_and(|n| n != 1) && lengths[1].is_some_and(|n| n != 1) {
                errors.push(argument_error(
                    "RM-CATALOG-RIDGE-VECTOR",
                    "input must be a vector",
                    argument,
                ));
            }
            shape.element_count()
        }
        ShapeFact::Shaped { .. } => {
            errors.push(argument_error(
                "RM-CATALOG-RIDGE-VECTOR",
                "input must be a vector",
                argument,
            ));
            None
        }
        ShapeFact::Ranked { rank } if *rank > 2 => {
            errors.push(argument_error(
                "RM-CATALOG-RIDGE-VECTOR",
                "input must be a vector",
                argument,
            ));
            None
        }
        _ => None,
    }
}

fn real_numeric(value: &ValueFact, argument: usize, errors: &mut Vec<InferenceDiagnostic>) {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        errors.push(argument_error(
            "RM-CATALOG-RIDGE-CLASS",
            "input must be real numeric or logical",
            argument,
        ));
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut errors = Vec::new();
    if !(3..=4).contains(&request.arguments.len()) {
        errors.push(argument_error(
            "RM-CATALOG-RIDGE-ARITY",
            "ridge requires y, X, k, and optional scaled",
            request.arguments.len(),
        ));
    }
    let response = request.arguments.first().and_then(|value| {
        real_numeric(value, 0, &mut errors);
        vector_length(&value.shape, 0, &mut errors)
    });
    let predictors = request.arguments.get(1).and_then(|value| {
        real_numeric(value, 1, &mut errors);
        match &value.shape {
            ShapeFact::Scalar => {
                if response.is_some_and(|n| n != 1) {
                    errors.push(argument_error(
                        "RM-CATALOG-RIDGE-ROWS",
                        "length(y) must match rows in X",
                        1,
                    ));
                }
                Some(1)
            }
            ShapeFact::Shaped { dims } if dims.len() == 2 => {
                if let DimensionFact::Known(rows) = dims[0] {
                    if response.is_some_and(|n| n != rows) {
                        errors.push(argument_error(
                            "RM-CATALOG-RIDGE-ROWS",
                            "length(y) must match rows in X",
                            1,
                        ));
                    }
                }
                match dims[1] {
                    DimensionFact::Known(cols) => Some(cols),
                    _ => None,
                }
            }
            ShapeFact::Shaped { .. } | ShapeFact::Ranked { rank: 3.. } => {
                errors.push(argument_error(
                    "RM-CATALOG-RIDGE-RANK",
                    "X must be a two-dimensional matrix",
                    1,
                ));
                None
            }
            _ => None,
        }
    });
    let parameters = request.arguments.get(2).and_then(|value| {
        real_numeric(value, 2, &mut errors);
        let len = vector_length(&value.shape, 2, &mut errors);
        if len == Some(0) {
            errors.push(argument_error(
                "RM-CATALOG-RIDGE-EMPTY-K",
                "k must not be empty",
                2,
            ));
        }
        len
    });
    if request
        .literals
        .literal_args
        .get(2)
        .is_some_and(invalid_known_parameter)
    {
        errors.push(argument_error(
            "RM-CATALOG-RIDGE-K",
            "k values must be nonnegative finite scalars",
            2,
        ));
    }
    let scaled = request.arguments.get(3).and_then(|value| {
        match &value.kind {
            ValueKindFact::Logical if value.storage == StorageFact::Scalar => {}
            ValueKindFact::Logical => errors.push(argument_error(
                "RM-CATALOG-RIDGE-SCALED",
                "scaled must be a scalar logical or numeric value",
                3,
            )),
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Unknown => {}
            _ => errors.push(argument_error(
                "RM-CATALOG-RIDGE-SCALED",
                "scaled must be zero or one",
                3,
            )),
        }
        if value.shape.element_count().is_some_and(|n| n != 1) {
            errors.push(argument_error(
                "RM-CATALOG-RIDGE-SCALED",
                "scaled must be scalar",
                3,
            ));
        }
        let known = request
            .literals
            .literal_bool_at(3)
            .map(RidgeScaling::from_bool)
            .or_else(|| {
                request
                    .literals
                    .numeric_at(3)
                    .and_then(RidgeScaling::from_float)
            });
        if request.literals.numeric_at(3).is_some() && known.is_none() {
            errors.push(argument_error(
                "RM-CATALOG-RIDGE-SCALED",
                "scaled must be zero or one",
                3,
            ));
        }
        known
    });
    let rows = match scaled {
        Some(RidgeScaling::Standardized) => predictors,
        Some(RidgeScaling::Original) => predictors.and_then(|n| n.checked_add(1)),
        None if request.arguments.len() == 3 => predictors,
        None => None,
    };
    let output = if errors.is_empty() {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![rows, parameters]),
            StorageFact::Dense,
        )
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(errors);
    result
}
