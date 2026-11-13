use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ISMETHOD-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.get(1) {
        if !matches!(input.kind, ValueKindFact::Unknown)
            && match input.kind {
                ValueKindFact::String => input.shape.element_count().is_some_and(|n| n != 1),
                ValueKindFact::Character => {
                    !crate::catalog::inference::possible_character_row(&input.shape)
                }
                _ => true,
            }
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISMETHOD-INPUT",
                "method name requires scalar row text",
                1,
            ));
        }
    }
    let outputs = vec![ValueFact::scalar(ValueKindFact::Logical)];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
