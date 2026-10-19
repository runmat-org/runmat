use crate::catalog::inference::{argument_error, known_invalid_text_scalar};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        result.diagnostics.push(argument_error(
            "RM-CALL-METHOD-ARITY",
            "call_method requires an object receiver and method name",
            request.arguments.len(),
        ));
        return result;
    }
    let receiver = &request.arguments[0];
    if !matches!(
        receiver.kind,
        ValueKindFact::Unknown | ValueKindFact::Object(_)
    ) {
        result.diagnostics.push(argument_error(
            "RM-CALL-METHOD-RECEIVER",
            "call_method requires an object receiver",
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
            "RM-CALL-METHOD-NAME",
            "method name must be scalar text",
            1,
        ));
    }
    result
}
