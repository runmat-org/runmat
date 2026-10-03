use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-ARITY",
            "filewrite requires a filename and data",
            0,
        ));
        return result;
    }
    if invalid_text(&request.arguments[0]) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-PATH",
            "filename must be scalar text",
            0,
        ));
    }
    if invalid_data(&request.arguments[1]) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-DATA",
            "data must be text, logical, or real numeric storage",
            1,
        ));
    }
    // The first trailing text may be a positional encoding; subsequent
    // name/value pairs are validated at runtime, including duplicate names.
    for (index, value) in request.arguments.iter().enumerate().skip(2) {
        if invalid_text(value) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-FILEWRITE-OPTION-TEXT",
                "filewrite options must be scalar text",
                index,
            ));
        }
    }
    result
}

fn invalid_text(value: &ValueFact) -> bool {
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

fn invalid_data(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String => false,
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Logical => false,
        _ => true,
    }
}
