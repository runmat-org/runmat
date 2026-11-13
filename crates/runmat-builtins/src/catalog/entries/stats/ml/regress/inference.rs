use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn real_input(value: &ValueFact, index: usize, errors: &mut Vec<InferenceDiagnostic>) {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        errors.push(argument_error(
            "RM-CATALOG-REGRESS-CLASS",
            "regression input must be real numeric or logical",
            index,
        ));
    }
}

fn extent(dimension: &DimensionFact) -> Option<usize> {
    match dimension {
        DimensionFact::Known(value) => Some(*value),
        _ => None,
    }
}

fn response_length(shape: &ShapeFact, errors: &mut Vec<InferenceDiagnostic>) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } if dims.len() == 1 => extent(&dims[0]),
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            if extent(&dims[0]).is_some_and(|n| n != 1) && extent(&dims[1]).is_some_and(|n| n != 1)
            {
                errors.push(argument_error(
                    "RM-CATALOG-REGRESS-Y",
                    "y must be a vector",
                    0,
                ));
            }
            shape.element_count()
        }
        ShapeFact::Shaped { .. } | ShapeFact::Ranked { rank: 3.. } => {
            errors.push(argument_error(
                "RM-CATALOG-REGRESS-Y",
                "y must be a vector",
                0,
            ));
            None
        }
        _ => None,
    }
}

fn design_shape(
    shape: &ShapeFact,
    errors: &mut Vec<InferenceDiagnostic>,
) -> (Option<usize>, Option<usize>) {
    match shape {
        ShapeFact::Scalar => (Some(1), Some(1)),
        ShapeFact::Shaped { dims } if dims.len() == 1 => (Some(1), extent(&dims[0])),
        ShapeFact::Shaped { dims } if dims.len() == 2 => (extent(&dims[0]), extent(&dims[1])),
        ShapeFact::Ranked { rank: 1 } => (Some(1), None),
        ShapeFact::Shaped { .. } | ShapeFact::Ranked { rank: 3.. } => {
            errors.push(argument_error(
                "RM-CATALOG-REGRESS-X",
                "X must be a two-dimensional design matrix",
                1,
            ));
            (None, None)
        }
        _ => (None, None),
    }
}

fn result(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut errors = Vec::new();
    // A discarded call admits numeric inputs and alpha without fitting.
    // Fit geometry and complete-observation requirements do not apply there.
    let fit_requested = request.outputs.requested.known_count() != Some(0);
    if !(2..=3).contains(&request.arguments.len()) {
        errors.push(argument_error(
            "RM-CATALOG-REGRESS-ARITY",
            "regress requires y, X, and optional alpha",
            request.arguments.len(),
        ));
    }
    let response = request.arguments.first().and_then(|value| {
        real_input(value, 0, &mut errors);
        if fit_requested {
            response_length(&value.shape, &mut errors)
        } else {
            None
        }
    });
    let (rows, columns) = request.arguments.get(1).map_or((None, None), |value| {
        real_input(value, 1, &mut errors);
        if fit_requested {
            design_shape(&value.shape, &mut errors)
        } else {
            (None, None)
        }
    });
    if rows.is_some() && response.is_some() && rows != response {
        errors.push(argument_error(
            "RM-CATALOG-REGRESS-ROWS",
            "length(y) must match rows in X",
            1,
        ));
    }
    if columns == Some(0) {
        errors.push(argument_error(
            "RM-CATALOG-REGRESS-COLUMNS",
            "X must contain at least one column",
            1,
        ));
    }
    if response == Some(0) || rows == Some(0) {
        errors.push(argument_error(
            "RM-CATALOG-REGRESS-EMPTY",
            "at least one complete observation is required",
            0,
        ));
    }
    if let Some(alpha) = request.arguments.get(2) {
        if !matches!(
            alpha.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Unknown
        ) {
            errors.push(argument_error(
                "RM-CATALOG-REGRESS-ALPHA-CLASS",
                "alpha must be a real numeric scalar",
                2,
            ));
        }
        if alpha.shape.element_count().is_some_and(|count| count != 1) {
            errors.push(argument_error(
                "RM-CATALOG-REGRESS-ALPHA",
                "alpha must be a numeric scalar",
                2,
            ));
        }
        if request
            .literals
            .numeric_at(2)
            .is_some_and(|value| !(0.0..1.0).contains(&value))
        {
            errors.push(argument_error(
                "RM-CATALOG-REGRESS-ALPHA-RANGE",
                "alpha must be between zero and one",
                2,
            ));
        }
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|n| n > 5)
    {
        errors.push(argument_error(
            "RM-CATALOG-REGRESS-OUTPUTS",
            "regress supports at most five outputs",
            request.arguments.len(),
        ));
    }
    // NaN-containing observations are removed at execution, so residual
    // lengths cannot be inferred solely from the visible input dimensions.
    let outputs = vec![
        result(ShapeFact::from(vec![columns, Some(1)])),
        result(ShapeFact::from(vec![columns, Some(2)])),
        result(ShapeFact::from(vec![None, Some(1)])),
        result(ShapeFact::from(vec![None, Some(2)])),
        result(ShapeFact::from(vec![Some(1), Some(4)])),
    ];
    let outputs = if errors.is_empty() {
        outputs
    } else {
        vec![ValueFact::unknown(DynamicReason::UnsupportedRepresentation); 5]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(errors);
    inferred
}
