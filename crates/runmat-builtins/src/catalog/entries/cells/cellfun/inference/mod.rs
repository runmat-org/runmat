mod callback;
mod invocation;
mod options;
mod output;
use crate::BuiltinCatalogEntry;
use runmat_types::{CallContract, CallInference, CallRequest, DynamicReason, ValueFact};
pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let plan = invocation::Plan::from_request(request);
    let mut diagnostics = plan.diagnostics;
    let count = request.outputs.requested.known_count().unwrap_or(1);
    let callback = callback::infer(request, &plan.cell_indices, &plan.extra_indices, count);
    diagnostics.extend(callback.diagnostics);
    let outputs = (0..count)
        .map(|index| {
            let element = callback
                .outputs
                .get(index)
                .cloned()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::UnresolvedCallable));
            match plan.uniform_output {
                Some(false) => output::nonuniform(element, plan.output_shape.clone()),
                Some(true) => output::uniform(element, plan.output_shape.clone()),
                None => {
                    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
                    output.shape = plan.output_shape.clone();
                    output
                }
            }
        })
        .collect();
    let mut contract = if request.outputs.requested.known_count().is_some() {
        CallContract::fixed(outputs)
    } else {
        let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
        contract.outputs = outputs;
        contract
    };
    contract.effects = entry.contract.effect_set();
    contract.effects.0.extend(callback.effects.0);
    contract.capabilities = entry.contract.capability_set();
    contract.capabilities.0.extend(callback.capabilities.0);
    let mut inference = runmat_types::infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}
