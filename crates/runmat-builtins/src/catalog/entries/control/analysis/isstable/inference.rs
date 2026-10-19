use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{standard, CallInference, CallRequest, ResidencyFact, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ISSTABLE-ARITY",
            "isstable expects exactly one transfer-function model",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(input) = request.arguments.first() {
        let valid_model = match &input.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) => class == &standard::TRANSFER_FUNCTION.owned(),
                None => object.class.is_none(),
            },
            _ => false,
        };
        if !valid_model
            || input.shape.element_count().is_some_and(|count| count != 1)
            || matches!(input.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISSTABLE-MODEL",
                "sys must be a scalar SISO transfer-function object",
                0,
            ));
        }
    }

    // A successfully parsed tf always returns one host logical scalar. The
    // pole calculation is runtime-dependent, but not the result's class/shape.
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Logical),
        diagnostics,
    )
}
