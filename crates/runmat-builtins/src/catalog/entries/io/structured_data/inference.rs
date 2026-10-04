//! XML input facts shared by the three identity-local inference rules.
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn scalar_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Character => !value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|dim| dim.is_some_and(|dim| dim != 1))
        }),
        ValueKindFact::String => !value.shape.is_definitely_nonscalar(),
        _ => false,
    }
}

pub(super) fn input(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request
        .arguments
        .get(index)
        .is_some_and(|fact| !scalar_text(fact))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-XML-TEXT",
            "XML text controls must be scalar text",
            index,
        ));
    }
}

pub(super) fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output: ValueFact,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
