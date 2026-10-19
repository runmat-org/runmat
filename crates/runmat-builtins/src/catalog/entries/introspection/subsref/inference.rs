use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-INTROSPECTION-ARITY",
            "expected 2 inputs",
            0,
        ));
    }

    if let Some(path) = request.arguments.get(1) {
        if !matches!(path.kind, ValueKindFact::Unknown | ValueKindFact::Struct(_))
            || path.shape.element_count() == Some(0)
        {
            diagnostics.push(argument_error(
                "RM-INDEXING-PATH",
                "expected a nonempty standard substruct path",
                1,
            ));
        }
    }

    let output = ValueFact::unknown(DynamicReason::DynamicDispatch);
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.variadic_output = Some(Box::new(output));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
