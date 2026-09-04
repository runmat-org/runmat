#[cfg(test)]
mod tests;

use crate::catalog::inference::support::facts::{
    materialize_preserving_sparse_storage, preserve_shape_as_dynamic,
};
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let prepared = match super::input::prepare(request, "realsqrt") {
        Ok(prepared) => prepared,
        Err(diagnostics) => {
            return finish_fixed(
                entry,
                request,
                ValueFact::unknown(DynamicReason::RuntimeValue),
                diagnostics,
            )
        }
    };
    let mut output = prepared.fact.clone();
    let mut diagnostics = prepared.diagnostics;
    match &prepared.fact.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double | NumericClass::Single,
            domain: NumericDomain::Real,
        }) => {
            if matches!(
                prepared.literal,
                Some(
                    super::literal::RootLiteralDomain::Negative
                        | super::literal::RootLiteralDomain::Complex
                )
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REALSQRT-DOMAIN",
                    "realsqrt input must be real and nonnegative",
                    0,
                ));
            }
            materialize_preserving_sparse_storage(&mut output);
            if matches!(output.storage, StorageFact::Sparse) {
                output.residency = ResidencyFact::Host;
            } else if matches!(output.residency, ResidencyFact::Device { .. }) {
                output.residency = ResidencyFact::Unknown;
            }
        }
        ValueKindFact::Unknown => preserve_shape_as_dynamic(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-REALSQRT-INPUT",
                "realsqrt requires real single or double input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
