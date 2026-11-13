use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Logical)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    if request.arguments.len() != 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-ISSTRING-ARITY",
            "isstring requires exactly one value",
            request.arguments.len(),
        ));
        result.outputs.iter_mut().for_each(|output| {
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        });
    }
    result
}
