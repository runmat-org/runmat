use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MUNLOCK-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        if !matches!(input.kind, ValueKindFact::Unknown)
            && (crate::catalog::inference::known_invalid_text_scalar(input)
                || !matches!(input.kind, ValueKindFact::String | ValueKindFact::Character))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-MUNLOCK-TEXT",
                "expected scalar row text",
                index,
            ));
        }
    }
    let outputs = vec![];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
