use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STR2FUNC-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(input.kind, ValueKindFact::Unknown)
            && (crate::catalog::inference::known_invalid_text_scalar(input)
                || !matches!(input.kind, ValueKindFact::String | ValueKindFact::Character))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STR2FUNC-INPUT",
                "str2func requires scalar row text",
                0,
            ));
        }
    }
    if request
        .literals
        .literal_args
        .first()
        .and_then(crate::catalog::inference::literal_text)
        .is_some_and(|text| {
            text.trim()
                .strip_prefix('@')
                .unwrap_or(text.trim())
                .trim()
                .is_empty()
        })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STR2FUNC-EMPTY",
            "function name must not be empty",
            0,
        ));
    }
    let outputs = vec![ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: vec![],
        parameters_complete: false,
        outputs: vec![],
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: vec![],
        captures_complete: false,
    }))];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
