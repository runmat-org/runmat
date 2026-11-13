use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn extent(dimension: &DimensionFact) -> Option<usize> {
    match dimension {
        DimensionFact::Known(value) => Some(*value),
        _ => None,
    }
}

fn vector_length(
    shape: &ShapeFact,
    index: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } if dims.len() == 1 => extent(&dims[0]),
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            if extent(&dims[0]).is_some_and(|n| n != 1) && extent(&dims[1]).is_some_and(|n| n != 1)
            {
                errors.push(argument_error(
                    "RM-CATALOG-NLPARCI-VECTOR",
                    "input must be a vector",
                    index,
                ));
            }
            shape.element_count()
        }
        ShapeFact::Shaped { .. } | ShapeFact::Ranked { rank: 3.. } => {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-VECTOR",
                "input must be a vector",
                index,
            ));
            None
        }
        _ => None,
    }
}

fn matrix_shape(
    shape: &ShapeFact,
    index: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> (Option<usize>, Option<usize>) {
    match shape {
        ShapeFact::Scalar => (Some(1), Some(1)),
        ShapeFact::Shaped { dims } if dims.len() == 1 => (extent(&dims[0]), Some(1)),
        ShapeFact::Shaped { dims } if dims.len() == 2 => (extent(&dims[0]), extent(&dims[1])),
        ShapeFact::Shaped { .. } | ShapeFact::Ranked { rank: 3.. } => {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-MATRIX",
                "input must be a two-dimensional matrix",
                index,
            ));
            (None, None)
        }
        _ => (None, None),
    }
}

fn floating(
    value: &ValueFact,
    index: usize,
    errors: &mut Vec<InferenceDiagnostic>,
) -> Option<NumericClass> {
    if value.storage == StorageFact::Sparse {
        errors.push(argument_error(
            "RM-CATALOG-NLPARCI-STORAGE",
            "input must be dense",
            index,
        ));
    }
    match &value.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }) => Some(NumericClass::Single),
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }) => Some(NumericClass::Double),
        ValueKindFact::Unknown => None,
        _ => {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-CLASS",
                "input must be real single or double",
                index,
            ));
            None
        }
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut errors = Vec::new();
    let count = request.arguments.len();
    if !matches!(count, 4 | 6) {
        errors.push(argument_error(
            "RM-CATALOG-NLPARCI-ARITY",
            "nlparci requires beta, residuals, Jacobian or Covar pair, and optional Alpha pair",
            count,
        ));
    }
    let coefficients = request.arguments.first().and_then(|beta| {
        floating(beta, 0, &mut errors);
        vector_length(&beta.shape, 0, &mut errors)
    });
    let observations = request.arguments.get(1).and_then(|residual| {
        floating(residual, 1, &mut errors);
        vector_length(&residual.shape, 1, &mut errors)
    });
    if coefficients == Some(0) || observations == Some(0) {
        errors.push(argument_error(
            "RM-CATALOG-NLPARCI-EMPTY",
            "beta and residuals must be nonempty",
            0,
        ));
    }
    let method = request.arguments.get(2).and_then(|value| {
        if !matches!(
            value.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-METHOD",
                "method must be scalar text",
                2,
            ));
        }
        if value.shape.element_count().is_some_and(|n| n != 1)
            && !matches!(value.kind, ValueKindFact::Character)
        {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-METHOD",
                "method must be scalar text",
                2,
            ));
        }
        request.literals.literal_string_at(2)
    });
    let method = method.as_deref().and_then(|value| {
        let method = super::NlparciSource::parse(value);
        if method.is_none() {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-METHOD",
                "method must be Jacobian or Covar",
                2,
            ));
        }
        method
    });
    if let Some(source) = request.arguments.get(3) {
        floating(source, 3, &mut errors);
        let (rows, columns) = matrix_shape(&source.shape, 3, &mut errors);
        if let Some(method) = method {
            let expected_rows = match method {
                super::NlparciSource::Jacobian => observations,
                super::NlparciSource::Covariance => coefficients,
            };
            if rows
                .zip(expected_rows)
                .is_some_and(|(actual, expected)| actual != expected)
                || columns
                    .zip(coefficients)
                    .is_some_and(|(actual, expected)| actual != expected)
            {
                errors.push(argument_error(
                    "RM-CATALOG-NLPARCI-SOURCE-SHAPE",
                    "Jacobian or Covar dimensions do not match beta and residuals",
                    3,
                ));
            }
        }
    }
    if count == 6 {
        if request.arguments.get(4).is_some_and(|value| {
            !matches!(
                value.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        }) {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-ALPHA-NAME",
                "Alpha option name must be scalar text",
                4,
            ));
        }
        if request
            .literals
            .literal_string_at(4)
            .is_some_and(|name| !name.eq_ignore_ascii_case("Alpha"))
        {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-ALPHA-NAME",
                "final option must be Alpha",
                4,
            ));
        }
        if let Some(alpha) = request.arguments.get(5) {
            floating(alpha, 5, &mut errors);
            if alpha.shape.element_count().is_some_and(|n| n != 1) {
                errors.push(argument_error(
                    "RM-CATALOG-NLPARCI-ALPHA",
                    "Alpha must be scalar",
                    5,
                ));
            }
        }
        if request
            .literals
            .numeric_at(5)
            .is_some_and(|alpha| !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0)
        {
            errors.push(argument_error(
                "RM-CATALOG-NLPARCI-ALPHA",
                "Alpha must be between zero and one",
                5,
            ));
        }
    }
    let output = if errors.is_empty() {
        let class = request.arguments.first().and_then(|beta| match &beta.kind {
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }) => Some(*class),
            _ => None,
        });
        match class {
            Some(class) => ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![coefficients, Some(2)]),
                StorageFact::Dense,
            ),
            None => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        }
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
