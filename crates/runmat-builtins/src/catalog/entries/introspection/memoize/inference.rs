use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MEMOIZE-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Callable(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MEMOIZE-INPUT",
                "memoize requires a function handle or closure",
                0,
            ));
        }
    }
    let mut outputs = vec![ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(crate::MEMOIZED_FUNCTION_CLASS.owned()),
        properties: std::collections::BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(true),
    }))];
    outputs[0].alias = AliasFact::Identity;
    outputs[0].mutation = MutationFact::HandleSemantics;
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
