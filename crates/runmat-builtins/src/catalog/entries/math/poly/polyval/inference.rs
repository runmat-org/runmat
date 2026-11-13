use super::super::PolyfitStatField;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYVAL-ARITY",
            "polyval requires coefficients, points, and at most S and mu",
            request.arguments.len().saturating_sub(1),
        ));
    }
    validate_coefficients(request.arguments.first(), &mut diagnostics);
    validate_points(request.arguments.get(1), &mut diagnostics);
    let count = request.outputs.requested.known_count();
    validate_stats(
        request.arguments.get(2),
        request.arguments.first(),
        count,
        &mut diagnostics,
    );
    validate_mu(request.arguments.get(3), &mut diagnostics);
    let primary = numeric_output(request.arguments.first(), request.arguments.get(1), false);
    let delta = numeric_output(request.arguments.first(), request.arguments.get(1), true);
    let outputs = match count {
        Some(0) => Vec::new(),
        Some(count) => (0..count)
            .map(|index| match index {
                0 => primary.clone(),
                1 => delta.clone(),
                _ => double_scalar(),
            })
            .collect(),
        None => vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = count.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn validate_coefficients(value: Option<&ValueFact>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = value else { return };
    if !numeric_or_logical_or_unknown(value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYVAL-COEFFICIENTS",
            "polynomial coefficients must be numeric or logical",
            0,
        ));
    }
    if let Some(dims) = value.shape.known_dims() {
        if dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYVAL-VECTOR",
                "polynomial coefficients must form a vector",
                0,
            ));
        }
    }
}

fn validate_points(value: Option<&ValueFact>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if value.is_some_and(|value| !numeric_or_logical_or_unknown(value)) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYVAL-POINTS",
            "evaluation points must be numeric or logical",
            1,
        ));
    }
}

fn validate_stats(
    value: Option<&ValueFact>,
    coefficients: Option<&ValueFact>,
    requested: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if requested.is_some_and(|count| count >= 2)
        && (value.is_none() || value.is_some_and(|value| value.shape.element_count() == Some(0)))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYVAL-DELTA-S",
            "prediction interval output requires a nonempty S structure",
            2,
        ));
    }
    let Some(value) = value else { return };
    if value.shape.element_count() == Some(0) {
        return;
    }
    let ValueKindFact::Struct(structure) = &value.kind else {
        if !matches!(value.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYVAL-S",
                "S must be a polyfit statistics structure or empty placeholder",
                2,
            ));
        }
        return;
    };
    validate_stat_fields(structure, coefficients, diagnostics);
}

fn validate_stat_fields(
    structure: &StructFact,
    coefficients: Option<&ValueFact>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    for field in PolyfitStatField::ALL {
        if structure.fields_complete && !structure.fields.contains_key(field.public_name()) {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYVAL-S-FIELD",
                format!("S requires field {}", field.public_name()),
                2,
            ));
        }
    }
    if let Some(r) = structure.fields.get(PolyfitStatField::R.public_name()) {
        let expected = coefficients.and_then(|value| value.shape.element_count());
        let dims = r.shape.known_dims();
        if dims.as_deref().is_some_and(|dims| {
            expected.is_some_and(|n| {
                dims.first()
                    .and_then(|dim| *dim)
                    .is_some_and(|rows| rows != n)
                    || dims
                        .get(1)
                        .and_then(|dim| *dim)
                        .is_some_and(|cols| cols != n)
            })
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYVAL-S-R",
                "S.R must be square with one row per coefficient",
                2,
            ));
        }
        if !numeric_or_logical_or_unknown(r) {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYVAL-S-R-CLASS",
                "S.R must be numeric or logical",
                2,
            ));
        }
    }
    for field in [
        PolyfitStatField::DegreesOfFreedom,
        PolyfitStatField::ResidualNorm,
    ] {
        if let Some(value) = structure.fields.get(field.public_name()) {
            if value.shape.element_count().is_some_and(|n| n != 1)
                || !matches!(
                    value.kind,
                    ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    }) | ValueKindFact::Logical
                        | ValueKindFact::Unknown
                )
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-POLYVAL-S-SCALAR",
                    format!("S.{} must be a real scalar", field.public_name()),
                    2,
                ));
            }
        }
    }
}

fn validate_mu(value: Option<&ValueFact>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = value else { return };
    if !numeric_or_logical_or_unknown(value) || value.shape.element_count().is_some_and(|n| n < 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYVAL-MU",
            "mu must contain at least two numeric values",
            3,
        ));
    }
}

fn numeric_or_logical_or_unknown(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    )
}

fn numeric_output(
    coefficients: Option<&ValueFact>,
    points: Option<&ValueFact>,
    delta: bool,
) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    let Some(points) = points else { return result };
    result.shape = points.shape.clone();
    result.storage = match points.shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    if let Some(coefficients) = coefficients {
        if matches!(coefficients.residency, ResidencyFact::Host)
            && matches!(points.residency, ResidencyFact::Host)
        {
            let domain = if delta || (real_or_logical(coefficients) && real_or_logical(points)) {
                Some(NumericDomain::Real)
            } else if complex_numeric(coefficients) || complex_numeric(points) {
                Some(NumericDomain::Complex)
            } else {
                None
            };
            if let Some(domain) = domain {
                result.kind = ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain,
                });
                result.residency = ResidencyFact::Host;
            }
        }
    }
    result
}

fn real_or_logical(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
    )
}

fn complex_numeric(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        })
    )
}

fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
