use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SQUARE-ARITY",
            "square expects times and optional duty cycle",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let output = match request.arguments.first() {
        Some(times) => sample_output(times, &mut diagnostics),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    if request.literals.integer_is_exact_binary64_at(0) == Some(false) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SQUARE-INTEGER-EXACTNESS",
            "square sample times cannot be converted to double exactly",
            0,
        ));
    }
    if let Some(duty) = request.arguments.get(1) {
        let scalar = !duty.shape.is_definitely_nonscalar();
        let real = matches!(
            duty.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        );
        if !scalar
            || !real
            || request
                .literals
                .numeric_at(1)
                .is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SQUARE-DUTY",
                "square duty must be a finite real scalar in [0, 100]",
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
            "RM-CATALOG-SQUARE-STORAGE",
            "square requires dense sample times",
            0,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let class = match times.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }) => NumericClass::Single,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => NumericClass::Double,
        ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-SQUARE-INPUT",
                "square expects real numeric sample times",
                0,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    };
    let storage = match times.shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        times.shape.clone(),
        storage,
    );
    output.residency = ResidencyFact::Host;
    output
}
