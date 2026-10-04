use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer_build(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    cuda: bool,
) -> CallInference {
    let mut contract = CallContract::fixed(Vec::new());
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    let name = if cuda { "mexcuda" } else { "mex" };
    if request.arguments.is_empty() {
        result.diagnostics.push(argument_error(
            "RM-MEX-SOURCE",
            format!("{name} requires a source file"),
            0,
        ));
    }
    // The existing runtime converts scalar numbers and logicals to text. The
    // shared build adapter, not this inference pass, interprets option syntax.
    for (index, fact) in request.arguments.iter().enumerate() {
        if definitely_unconvertible_to_build_text(fact) {
            result.diagnostics.push(argument_error(
                "RM-MEX-ARGUMENT",
                format!("{name} argument cannot be converted to build text"),
                index,
            ));
        }
    }
    result
}

pub(super) fn infer_extension(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::String)]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !request.arguments.is_empty() {
        result.diagnostics.push(argument_error(
            "RM-MEXEXT-ARITY",
            "mexext accepts no inputs",
            0,
        ));
    }
    result
}

fn definitely_unconvertible_to_build_text(fact: &ValueFact) -> bool {
    if matches!(
        fact.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
    ) {
        return true;
    }
    match &fact.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => fact.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => {
            fact.shape.rank().is_some_and(|rank| rank > 2)
                || fact.shape.known_dims().is_some_and(|dims| {
                    dims.first()
                        .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                        || dims
                            .iter()
                            .skip(2)
                            .any(|dim| dim.is_some_and(|dim| dim != 1))
                })
        }
        ValueKindFact::Logical => {
            matches!(fact.storage, StorageFact::Dense | StorageFact::Sparse)
                || fact.shape.is_definitely_nonscalar()
        }
        ValueKindFact::Numeric(numeric) => {
            numeric.domain == NumericDomain::Complex
                || matches!(fact.storage, StorageFact::Dense | StorageFact::Sparse)
                || fact.shape.is_definitely_nonscalar()
        }
        _ => true,
    }
}
