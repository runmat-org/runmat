use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest, LiteralValue, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = runmat_types::CallContract::fixed(vec![]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = runmat_types::infer_call(&contract, request);
    if request.arguments.len() > 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FORMAT-ARITY",
            "format accepts at most one mode",
            1,
        ));
    }
    if let Some(value) = request.arguments.first() {
        if definitely_nontext(value) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-FORMAT-TEXT",
                "format mode must be a text scalar",
                0,
            ));
        } else if let Some(LiteralValue::String(text) | LiteralValue::Character(text)) =
            request.literals.literal_args.first()
        {
            if super::FormatCommand::decode(text).is_none() {
                result.diagnostics.push(argument_error(
                    "RM-CATALOG-FORMAT-MODE",
                    "unknown format mode",
                    0,
                ));
            }
        }
    }
    result
}

fn definitely_nontext(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .copied()
                .flatten()
                .is_some_and(|rows| rows != 1)
                || dims
                    .iter()
                    .skip(2)
                    .any(|extent| extent.is_some_and(|n| n > 1))
        }),
        ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}
