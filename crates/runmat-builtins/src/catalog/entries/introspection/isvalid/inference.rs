use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ISVALID-ARITY",
            "isvalid requires one handle or listener",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.first() {
        let admissible = match &input.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                object.handle_semantics != Some(false) && !input.shape.is_definitely_nonscalar()
            }
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISVALID-INPUT",
                "isvalid requires a handle object or listener",
                0,
            ));
        }
    }
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Logical)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    if !diagnostics.is_empty() {
        result.outputs.iter_mut().for_each(|output| {
            *output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        });
    }
    result.diagnostics.extend(diagnostics);
    result
}
