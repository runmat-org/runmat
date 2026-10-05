use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-FUNCTIONS-ARITY",
            "expected one function handle",
            0,
        ));
    }
    if let Some(input) = request.arguments.first() {
        if matches!(
            input.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-FUNCTIONS-RESIDENCY",
                "function handle must be host-resident",
                0,
            ));
        }
        if !matches!(
            input.kind,
            ValueKindFact::Callable(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-FUNCTIONS-HANDLE",
                "expected a function handle",
                0,
            ));
        }
    }
    let text = ValueFact::scalar(ValueKindFact::String);
    let fields = BTreeMap::from([
        ("function".to_string(), text.clone()),
        ("type".to_string(), text.clone()),
        ("file".to_string(), text),
    ]);
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
