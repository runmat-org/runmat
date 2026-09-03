use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericDomain, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [input] => match input.kind {
            ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
                if !input.is_scalar() && input.shape.element_count().is_some() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-PRIMES-SCALAR",
                        "primes requires a scalar input",
                        0,
                    ));
                }
                ValueFact::proven(
                    input.kind.clone(),
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                )
            }
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PRIMES-INPUT",
                    "primes requires a real numeric scalar",
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        },
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PRIMES-ARITY",
                "primes requires exactly one input",
                request.arguments.len().min(1),
            ));
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}
