mod callback;
mod options;
mod output;
mod validation;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallContract, CallInference, CallRequest, DynamicReason};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let options = options::Options::parse(request);
    let mut diagnostics = options.diagnostics;
    validation::validate(request, &mut diagnostics);

    let requested = request.outputs.requested.known_count().unwrap_or(1);
    let inferred = callback::infer(request, requested);
    diagnostics.extend(inferred.diagnostics);

    let templates = output::collect(&inferred.fields, requested, options.uniform);
    let mut contract = if request.outputs.requested.known_count().is_some() {
        CallContract::fixed(templates)
    } else {
        let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
        contract.outputs = templates;
        contract
    };
    contract.effects = entry.contract.effect_set();
    contract.effects.0.extend(inferred.effects.0);
    contract.capabilities = entry.contract.capability_set();
    contract.capabilities.0.extend(inferred.capabilities.0);
    let mut result = runmat_types::infer_call(&contract, request);
    diagnostics.append(&mut result.diagnostics);
    result.diagnostics = diagnostics;
    result
}
