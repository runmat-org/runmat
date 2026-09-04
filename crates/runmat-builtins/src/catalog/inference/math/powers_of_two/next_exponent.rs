use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, StorageFact, ValueFact,
    ValueKindFact,
};

use super::super::super::support::facts::{materialize, preserve_shape_as_dynamic};
use super::super::super::{argument_error, finish_fixed, numeric_kind};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-NEXTPOW2-ARITY",
            "nextpow2 requires exactly one input",
            request.arguments.len().min(1),
        ));
    }
    let Some(input) = request.arguments.first() else {
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let mut output = input.clone();
    match output.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            if matches!(input.storage, StorageFact::Sparse) {
                diagnostics.push(input_error("nextpow2 does not accept sparse input"));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            } else {
                materialize(&mut output);
            }
        }
        ValueKindFact::Logical => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize(&mut output);
        }
        ValueKindFact::Unknown => preserve_shape_as_dynamic(&mut output),
        _ => {
            diagnostics.push(input_error(
                "nextpow2 requires real numeric or logical input",
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn input_error(message: &'static str) -> runmat_types::InferenceDiagnostic {
    argument_error("RM-CATALOG-NEXTPOW2-INPUT", message, 0)
}
