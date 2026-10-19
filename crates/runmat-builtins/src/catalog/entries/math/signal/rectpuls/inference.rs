use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RECTPULS-ARITY",
            "rectpuls expects sample times and an optional width",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let output = match request.arguments.first() {
        Some(times) => sample_output(times, &mut diagnostics),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if request.literals.integer_is_exact_binary64_at(0) == Some(false) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RECTPULS-INTEGER-EXACTNESS",
            "rectpuls sample times cannot be converted to double exactly",
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
                "RM-CATALOG-RECTPULS-WIDTH",
                "rectpuls width must be a positive real scalar",
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
                "RM-CATALOG-RECTPULS-WIDTH",
                "rectpuls width must be positive and finite",
                1,
            ));
        }
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
            "RM-CATALOG-RECTPULS-STORAGE",
            "rectpuls requires dense sample times",
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
                "RM-CATALOG-RECTPULS-INPUT",
                "rectpuls expects real numeric sample times",
                0,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    let storage = if times.shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if times.shape.element_count().is_some() {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        times.shape.clone(),
        storage,
    );
    output.residency = ResidencyFact::Unknown;
    output
}
