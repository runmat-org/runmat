use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let text = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![text]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !(1..=3).contains(&request.arguments.len()) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEREAD-ARITY",
            "fileread requires a filename and at most one encoding option",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate().take(3) {
        if definitely_invalid_text(input) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-FILEREAD-TEXT",
                "filename and encoding controls must be scalar text",
                index,
            ));
        }
    }
    if request.arguments.len() == 3
        && request
            .literals
            .literal_string_at(1)
            .is_some_and(|text| !text.eq_ignore_ascii_case("Encoding"))
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEREAD-OPTION",
            "the only named option is Encoding",
            1,
        ));
    }
    result
}

fn definitely_invalid_text(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|dim| dim.is_some_and(|dim| dim != 1))
        }),
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}
