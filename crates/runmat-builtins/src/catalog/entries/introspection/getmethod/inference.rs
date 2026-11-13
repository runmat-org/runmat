use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GETMETHOD-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.get(1) {
        if !matches!(input.kind, ValueKindFact::Unknown)
            && (crate::catalog::inference::known_invalid_text_scalar(input)
                || !matches!(input.kind, ValueKindFact::String | ValueKindFact::Character))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GETMETHOD-INPUT",
                "getmethod requires scalar row text",
                1,
            ));
        }
    }
    if request
        .literals
        .literal_args
        .get(1)
        .and_then(crate::catalog::inference::literal_text)
        .is_some_and(|text| text.trim().is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GETMETHOD-EMPTY",
            "function name must not be empty",
            0,
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        if !matches!(
            receiver.kind,
            ValueKindFact::Unknown | ValueKindFact::Object(_) | ValueKindFact::ClassReference(_)
        ) || matches!(
            receiver.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GETMETHOD-RECEIVER",
                "getmethod requires a host object, handle object, or class reference",
                0,
            ));
        }
    }
    if let Some(name) = request.arguments.get(1) {
        if matches!(
            name.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GETMETHOD-NAME",
                "method name must be host text",
                1,
            ));
        }
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
