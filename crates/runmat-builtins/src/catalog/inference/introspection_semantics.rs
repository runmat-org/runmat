use super::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{CallContract, CallInference, CallRequest, DynamicReason, ValueKindFact};

pub(super) fn infer_feval(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut contract = match request.arguments.first().map(|argument| &argument.kind) {
        Some(ValueKindFact::Callable(callable)) => {
            callable.call_contract(DynamicReason::RuntimeValue)
        }
        Some(_) => CallContract::dynamic(DynamicReason::RuntimeValue),
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEVAL-ARITY",
                "feval requires a function target",
                0,
            ));
            CallContract::dynamic(DynamicReason::RuntimeValue)
        }
    };
    contract.effects = entry.contract.effect_set();
    contract
        .capabilities
        .0
        .extend(entry.contract.capability_set().0);
    let mut inference = runmat_types::infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}
