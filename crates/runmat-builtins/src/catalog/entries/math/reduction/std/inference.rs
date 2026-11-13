use super::super::support::{admitted_std_controls, inferred_dispersion_shape};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.first() {
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.STD.INVALID_ARGUMENT",
                "std requires input data",
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        Some(input) => infer_output(input, request, &mut diagnostics),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn infer_output(
    input: &ValueFact,
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    if input.storage == StorageFact::Sparse {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let class = match &input.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.STD.COMPLEX_UNSUPPORTED",
                "std does not support complex input",
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) if numeric.class.integer_class().is_some() => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.STD.INVALID_INPUT",
                "std does not support integer input data",
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => numeric.class,
        ValueKindFact::Logical => NumericClass::Double,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let Some(controls) = admitted_std_controls(request.arguments.len(), &request.literals) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let shape = inferred_dispersion_shape(&input.shape, &controls);
    let storage = if input.residency == ResidencyFact::Host {
        match shape.element_count() {
            // Single has no immediate scalar Value; a one-element result
            // remains typed tensor storage.
            Some(1) if class == NumericClass::Single => StorageFact::Dense,
            Some(1) => StorageFact::Scalar,
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        }
    } else {
        StorageFact::Unknown
    };
    let mut result = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    if input.residency != ResidencyFact::Host {
        result.residency = ResidencyFact::Unknown;
    }
    result
}
