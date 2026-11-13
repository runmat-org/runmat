use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn double(shape: ShapeFact, input: Option<&ValueFact>) -> ValueFact {
    let storage = if input.is_some_and(|value| !matches!(value.residency, ResidencyFact::Host)) {
        StorageFact::Unknown
    } else {
        match shape.element_count() {
            Some(1) => StorageFact::Scalar,
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        }
    };
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    if input.is_some_and(|value| !matches!(value.residency, ResidencyFact::Host)) {
        fact.residency = ResidencyFact::Unknown;
    }
    fact
}

pub(super) fn finish(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    mut contract: CallContract,
    mut diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}
