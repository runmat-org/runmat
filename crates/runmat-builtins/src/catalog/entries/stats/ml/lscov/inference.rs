use super::LscovAlgorithm;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn matrix_dimensions(
    shape: &ShapeFact,
    index: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> [Option<usize>; 2] {
    match shape {
        ShapeFact::Scalar => [Some(1), Some(1)],
        ShapeFact::Shaped { dims } if dims.len() == 1 => [
            Some(1),
            match dims[0] {
                DimensionFact::Known(n) => Some(n),
                _ => None,
            },
        ],
        ShapeFact::Shaped { dims } if dims.len() == 2 => [
            match dims[0] {
                DimensionFact::Known(n) => Some(n),
                _ => None,
            },
            match dims[1] {
                DimensionFact::Known(n) => Some(n),
                _ => None,
            },
        ],
        ShapeFact::Ranked { rank: 1 } => [Some(1), None],
        ShapeFact::Ranked { rank: 2 } | ShapeFact::Unknown => [None, None],
        _ => {
            errors.push(argument_error(
                "RM-CATALOG-LSCOV-RANK",
                "input must be a two-dimensional matrix",
                index,
            ));
            [None, None]
        }
    }
}

fn numerical_domain(
    value: &ValueFact,
    index: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> Option<NumericDomain> {
    match value.kind {
        ValueKindFact::Numeric(fact) => Some(fact.domain),
        ValueKindFact::Logical => Some(NumericDomain::Real),
        ValueKindFact::Unknown => None,
        _ => {
            errors.push(argument_error(
                "RM-CATALOG-LSCOV-CLASS",
                "input must be numeric or logical",
                index,
            ));
            None
        }
    }
}

fn result(kind: Option<NumericDomain>, shape: ShapeFact) -> ValueFact {
    match kind {
        Some(domain) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain,
            }),
            shape,
            StorageFact::Dense,
        ),
        None => {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.shape = shape;
            fact.storage = StorageFact::Dense;
            fact
        }
    }
}

fn algorithm_literal(request: &CallRequest, index: usize, errors: &mut Vec<InferenceDiagnostic>) {
    if request.arguments.get(index).is_some_and(|value| {
        !matches!(
            value.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        )
    }) {
        errors.push(argument_error(
            "RM-CATALOG-LSCOV-ALG",
            "algorithm must be text",
            index,
        ));
    }
    if let Some(value) = request.arguments.get(index) {
        let invalid_shape = match (&value.kind, &value.shape) {
            (ValueKindFact::String, shape) => shape.element_count().is_some_and(|count| count != 1),
            (ValueKindFact::Character, ShapeFact::Shaped { dims }) if dims.len() >= 2 => {
                matches!(&dims[0], DimensionFact::Known(rows) if *rows != 1)
            }
            _ => false,
        };
        if invalid_shape {
            errors.push(argument_error(
                "RM-CATALOG-LSCOV-ALG-SHAPE",
                "algorithm must be a string scalar or character row",
                index,
            ));
        }
    }
    if let Some(keyword) = request.literals.literal_string_at(index) {
        if LscovAlgorithm::from_public_keyword(&keyword).is_none() {
            errors.push(argument_error(
                "RM-CATALOG-LSCOV-ALG",
                "algorithm must be orth or chol",
                index,
            ));
        }
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut errors = Vec::new();
    if !(2..=4).contains(&request.arguments.len()) {
        errors.push(argument_error(
            "RM-CATALOG-LSCOV-ARITY",
            "lscov requires A, B, and up to two optional arguments",
            request.arguments.len(),
        ));
    }
    let a = request.arguments.first().map(|value| {
        let domain = numerical_domain(value, 0, &mut errors);
        let dimensions = matrix_dimensions(&value.shape, 0, &mut errors);
        if dimensions.contains(&Some(0)) {
            errors.push(argument_error(
                "RM-CATALOG-LSCOV-EMPTY",
                "A must not be empty",
                0,
            ));
        }
        (domain, dimensions)
    });
    let b = request.arguments.get(1).map(|value| {
        let domain = numerical_domain(value, 1, &mut errors);
        let dimensions = matrix_dimensions(&value.shape, 1, &mut errors);
        (domain, dimensions)
    });
    let a_rows = a.as_ref().and_then(|(_, dimensions)| dimensions[0]);
    let predictors = a.as_ref().and_then(|(_, dimensions)| dimensions[1]);
    let b_dims = b
        .as_ref()
        .map(|(_, dimensions)| *dimensions)
        .unwrap_or([None, None]);
    let b_count = request
        .arguments
        .get(1)
        .and_then(|value| value.shape.element_count());
    let known_vector = b_dims[0] == Some(1) || b_dims[1] == Some(1);
    let response_columns = if known_vector && a_rows.is_some() && b_count == a_rows {
        Some(1)
    } else if b_dims[0] == a_rows && a_rows.is_some() {
        b_dims[1]
    } else {
        if a_rows.is_some()
            && b_dims[0].is_some()
            && b_count.is_some()
            && !(known_vector && b_count == a_rows)
        {
            errors.push(argument_error(
                "RM-CATALOG-LSCOV-ROWS",
                "B must have one row per observation",
                1,
            ));
        }
        None
    };
    if request.arguments.len() >= 3 {
        let third_is_algorithm = request.arguments.get(2).is_some_and(|value| {
            matches!(value.kind, ValueKindFact::String | ValueKindFact::Character)
        }) || request.literals.literal_string_at(2).is_some();
        if request.arguments.len() == 3 && third_is_algorithm {
            algorithm_literal(request, 2, &mut errors);
        } else if let Some(weight) = request.arguments.get(2) {
            match weight.kind {
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
                | ValueKindFact::Logical
                | ValueKindFact::Unknown => {}
                _ => errors.push(argument_error(
                    "RM-CATALOG-LSCOV-WEIGHT",
                    "V must be real numeric or logical",
                    2,
                )),
            }
            let known_empty = weight.shape.element_count() == Some(0);
            let dimensions = if known_empty {
                [None, None]
            } else {
                matrix_dimensions(&weight.shape, 2, &mut errors)
            };
            if !known_empty && a_rows.is_some() && dimensions.iter().all(Option::is_some) {
                let count = weight.shape.element_count();
                let vector = dimensions[0] == Some(1) || dimensions[1] == Some(1);
                let scalar_or_empty = count == Some(1) || count == Some(0);
                let valid_weight_shape = scalar_or_empty
                    || dimensions == [a_rows, a_rows]
                    || (vector && count == a_rows);
                if !valid_weight_shape {
                    errors.push(argument_error(
                        "RM-CATALOG-LSCOV-WEIGHT-SHAPE",
                        "V must be a weight vector or matching square covariance",
                        2,
                    ));
                }
            }
        }
    }
    if request.arguments.len() == 4 {
        algorithm_literal(request, 3, &mut errors);
    }
    if request.outputs.requested.known_count() == Some(4)
        && response_columns.is_some_and(|count| count != 1)
    {
        errors.push(argument_error(
            "RM-CATALOG-LSCOV-S",
            "fourth output requires a vector response",
            1,
        ));
    }
    let coefficient_domain = match (
        a.and_then(|(domain, _)| domain),
        b.and_then(|(domain, _)| domain),
    ) {
        (Some(NumericDomain::Complex), _) | (_, Some(NumericDomain::Complex)) => {
            Some(NumericDomain::Complex)
        }
        (Some(NumericDomain::Real), Some(NumericDomain::Real)) => Some(NumericDomain::Real),
        _ => None,
    };
    let coefficient_shape = ShapeFact::from(vec![predictors, response_columns]);
    let outputs = vec![
        result(coefficient_domain, coefficient_shape.clone()),
        result(Some(NumericDomain::Real), coefficient_shape),
        result(
            Some(NumericDomain::Real),
            ShapeFact::from(vec![Some(1), response_columns]),
        ),
        result(
            coefficient_domain,
            ShapeFact::from(vec![predictors, predictors]),
        ),
    ];
    let outputs = if errors.is_empty() {
        outputs
    } else {
        vec![ValueFact::unknown(DynamicReason::UnsupportedRepresentation); 4]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(errors);
    inferred
}
