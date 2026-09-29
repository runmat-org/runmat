use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut contract = match request.arguments.first().map(|argument| &argument.kind) {
        Some(ValueKindFact::Callable(callable)) => {
            callable.call_contract(DynamicReason::RuntimeValue)
        }
        Some(ValueKindFact::Unknown) => CallContract::dynamic(DynamicReason::RuntimeValue),
        Some(_) => {
            diagnostics.push(argument_error(
                "RM-DL-DLFEVAL-FUNCTION",
                "dlfeval requires a function handle or closure, not text or a callable object",
                0,
            ));
            CallContract::dynamic(DynamicReason::RuntimeValue)
        }
        None => {
            diagnostics.push(argument_error(
                "RM-DL-DLFEVAL-ARITY",
                "dlfeval requires a function handle",
                0,
            ));
            CallContract::dynamic(DynamicReason::RuntimeValue)
        }
    };
    contract.effects.0.extend(entry.contract.effect_set().0);
    contract
        .capabilities
        .0
        .extend(entry.contract.capability_set().0);
    let mut inferred = infer_call(&contract, request);
    diagnostics.append(&mut inferred.diagnostics);
    inferred.diagnostics = diagnostics;
    inferred
}
