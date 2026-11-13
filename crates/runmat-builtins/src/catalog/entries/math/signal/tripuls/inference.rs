use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRIPULS-ARITY",
            "tripuls expects sample times and optional width and skew",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let output = match request.arguments.first() {
        Some(times) => sample_output(times, &mut diagnostics),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if request.literals.integer_is_exact_binary64_at(0) == Some(false) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRIPULS-INTEGER-EXACTNESS",
            "tripuls sample times cannot be converted to double exactly",
            0,
        ));
    }
    if let Some(width) = request.arguments.get(1) {
        let scalar = !width.shape.is_definitely_nonscalar();
        let real = matches!(
            width.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        );
        if !scalar || !real {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIPULS-WIDTH",
                "tripuls width must be a positive real scalar",
                1,
            ));
        }
        if request
            .literals
            .numeric_at(1)
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
            || request.literals.integer_is_exact_binary64_at(1) == Some(false)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIPULS-WIDTH",
                "tripuls width must be positive and finite",
                1,
            ));
        }
    }
    if let Some(skew) = request.arguments.get(2) {
        validate_skew(skew, request.literals.numeric_at(2), &mut diagnostics);
    }
    let output = if diagnostics.is_empty() {
        output
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn sample_output(times: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) -> ValueFact {
    if matches!(times.storage, StorageFact::Sparse | StorageFact::Opaque) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRIPULS-STORAGE",
            "tripuls requires dense sample times",
            0,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    match times.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => {}
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIPULS-INPUT",
                "tripuls expects real numeric sample times",
                0,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    let output_class = match times.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        }) => NumericClass::Single,
        _ => NumericClass::Double,
    };
    let storage = if times.shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if times.shape.element_count().is_some() {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: output_class,
            domain: NumericDomain::Real,
        }),
        times.shape.clone(),
        storage,
    );
    output.residency = ResidencyFact::Unknown;
    output
}

fn validate_skew(
    value: &ValueFact,
    literal: Option<f64>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let scalar = !value.shape.is_definitely_nonscalar();
    let real = matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    );
    if !scalar
        || !real
        || literal.is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRIPULS-SKEW",
            "tripuls skew must be a finite real scalar in [-1, 1]",
            2,
        ));
    }
}
