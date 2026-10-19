use crate::catalog::inference::{argument_error, known_invalid_text_scalar};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() < 3 {
        result.diagnostics.push(argument_error(
            "RM-BOUND-METHOD-ARITY",
            "bound method invocation requires receiver, method, and scope",
            request.arguments.len(),
        ));
        return result;
    }
    if !matches!(
        request.arguments[0].kind,
        ValueKindFact::Unknown | ValueKindFact::Object(_)
    ) {
        result.diagnostics.push(argument_error(
            "RM-BOUND-METHOD-RECEIVER",
            "bound method requires an object receiver",
            0,
        ));
    }
    let method = &request.arguments[1];
    if !matches!(
        method.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) || known_invalid_text_scalar(method)
    {
        result.diagnostics.push(argument_error(
            "RM-BOUND-METHOD-NAME",
            "method name must be scalar text",
            1,
        ));
    }
    let scope = &request.arguments[2];
    if !matches!(scope.kind, ValueKindFact::Unknown | ValueKindFact::String)
        || known_invalid_text_scalar(scope)
    {
        result.diagnostics.push(argument_error(
            "RM-BOUND-METHOD-SCOPE",
            "scope must be a string scalar",
            2,
        ));
    }
    result
}
