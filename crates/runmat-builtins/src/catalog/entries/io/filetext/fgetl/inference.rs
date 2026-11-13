use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // A successful read is a character row, while an EOF read is double -1.
    // The fact model has no disjoint union kind, so this value stays dynamic.
    let mut contract = CallContract::fixed(vec![ValueFact::unknown(
        DynamicReason::ConflictingControlFlow,
    )]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(double()));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FGETL-ARITY",
            "fgetl requires one file identifier",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(definitely_invalid_fid)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FGETL-FID",
            "file identifier must be a real numeric scalar",
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
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
