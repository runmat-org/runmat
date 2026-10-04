use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer_with(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn infer_dynamic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn owned_pointer() -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Foreign(ForeignFact {
            family: "native-ffi".into(),
            type_name: None,
            type_version: None,
            ownership: ForeignOwnershipFact::Owned,
            affinity: ForeignAffinityFact::OriginThread,
            lifetime: ForeignLifetimeFact::Session,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
}

pub(super) fn text_at(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if let Some(fact) = request.arguments.get(index) {
        if !matches!(
            fact.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-NATIVE-FFI-TEXT",
                "native library path and alias arguments must be text",
                index,
            ));
        }
    }
}
