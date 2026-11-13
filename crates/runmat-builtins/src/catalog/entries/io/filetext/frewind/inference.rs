use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(Vec::new());
    // The runtime's legacy adapter returns an empty OutputList even when a
    // caller asks for a slot. Do not diagnose that request as an arity error;
    // the adapter, not this catalog, owns the eventual assignment behavior.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(
        DynamicReason::UnsupportedRepresentation,
    )));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FREWIND-ARITY",
            "frewind requires one file identifier",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(definitely_invalid_fid)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FREWIND-FID",
            "file identifier must be a real numeric or logical scalar",
            0,
        ));
    }
    result
}

fn definitely_invalid_fid(value: &ValueFact) -> bool {
    if value.shape.is_definitely_nonscalar() {
        return true;
    }
    match &value.kind {
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Logical | ValueKindFact::Unknown => false,
        _ => true,
    }
}
