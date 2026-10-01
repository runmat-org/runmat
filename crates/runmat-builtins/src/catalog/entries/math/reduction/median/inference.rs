use super::super::support::{admitted_median_controls, inferred_median_shape};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.first() {
        None => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.MEDIAN.INVALID_ARGUMENT",
                "median requires numeric or logical input data",
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        Some(input) => infer_output(input, request),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn infer_output(input: &ValueFact, request: &CallRequest) -> ValueFact {
    if input.storage == StorageFact::Sparse {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let kind = match input.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            ValueKindFact::Numeric(numeric)
        }
        ValueKindFact::Logical => ValueKindFact::Logical,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let Some(controls) = admitted_median_controls(request.arguments.len(), &request.literals)
    else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let shape = inferred_median_shape(&input.shape, &controls);
    // An empty selected slice returns a floating NaN, even when the ordinary
    // integer path retains the input class. No class claim is sound here.
    if input.shape.element_count() == Some(0) {
        let mut dynamic = ValueFact::unknown(DynamicReason::RuntimeValue);
        dynamic.shape = shape;
        return dynamic;
    }
    let storage = if input.residency == ResidencyFact::Host {
        match (shape.element_count(), &kind) {
            (Some(1), ValueKindFact::Numeric(numeric)) if numeric.class == NumericClass::Double => {
                StorageFact::Scalar
            }
            (Some(1), ValueKindFact::Logical) => StorageFact::Scalar,
            (Some(1), _) => StorageFact::Unknown,
            (Some(_), _) => StorageFact::Dense,
            (None, _) => StorageFact::Unknown,
        }
    } else {
        StorageFact::Unknown
    };
    let mut result = ValueFact::proven(kind, shape, storage);
    if input.residency != ResidencyFact::Host {
        result.residency = ResidencyFact::Unknown;
    }
    result
}
