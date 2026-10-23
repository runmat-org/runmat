//! Shared chart-result facts, not a second parser or graphics-handle registry.
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) enum Handles {
    Scalar,
    Series,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    minimum_inputs: usize,
    handles: Handles,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < minimum_inputs {
        diagnostics.push(argument_error(
            "RM-CATALOG-COORDINATE-ARITY",
            "insufficient chart inputs",
            request.arguments.len(),
        ));
    }
    // Numeric scalar coordinates are not statically assumed to be axes aliases.
    // Shape-dependent series expansion and value-selected properties belong to
    // Runtime; only the scalar-result identities have a proven scalar shape.
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        match handles {
            Handles::Scalar => ShapeFact::Scalar,
            Handles::Series => ShapeFact::Unknown,
        },
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    if !diagnostics.is_empty() {
        output = ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}
