use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(request.arguments.clone());
    if let [single] = request.arguments.as_slice() {
        contract.variadic_output = Some(Box::new(single.clone()));
        contract.maximum_outputs = None;
    }
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 1
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|n| n != request.arguments.len())
    {
        result.diagnostics.push(argument_error(
            "RM-DEAL-COUNT",
            "deal input and output counts must match unless there is one input",
            0,
        ));
    }
    result
}
