use crate::catalog::inference::{argument_error, known_invalid_text_scalar};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    let Some(name) = request.arguments.first() else {
        result.diagnostics.push(argument_error(
            "RM-BUILTIN-NAME-MISSING",
            "builtin requires a function name",
            0,
        ));
        return result;
    };
    if !matches!(
        name.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) || known_invalid_text_scalar(name)
    {
        result.diagnostics.push(argument_error(
            "RM-BUILTIN-NAME-INVALID",
            "builtin name must be scalar text",
            0,
        ));
        return result;
    }
    // The target's admission and output facts depend on runtime resolution,
    // even when source text is known. Preserve declared target effects and
    // capabilities without pretending its output or residency is known here.
    if let Some(target_name) = request.literals.literal_string_at(0) {
        if target_name != "builtin" {
            if let Some(target) = crate::builtin_catalog_entry_by_name(&target_name) {
                result.effects.0.extend(target.contract.effect_set().0);
                result
                    .capabilities
                    .0
                    .extend(target.contract.capability_set().0);
            }
        }
    }
    result
}
