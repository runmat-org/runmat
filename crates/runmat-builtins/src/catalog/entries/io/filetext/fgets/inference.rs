use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // A successful line is a character row, but EOF produces double -1.
    // Terminators likewise alternate between a numeric row and the sentinel.
    let mixed = ValueFact::unknown(DynamicReason::ConflictingControlFlow);
    let mut contract = CallContract::fixed(vec![mixed.clone(), mixed]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(double()));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !(1..=2).contains(&request.arguments.len()) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FGETS-ARITY",
            "fgets requires a file identifier and optional nchar",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate().take(2) {
        if definitely_invalid_scalar(input) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-FGETS-SCALAR",
                "file identifier and nchar must be real numeric scalars",
                index,
            ));
        }
    }
    result
}

fn definitely_invalid_scalar(value: &ValueFact) -> bool {
    value.shape.is_definitely_nonscalar()
        || match &value.kind {
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
