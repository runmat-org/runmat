use super::admission::AdmittedCall;
use crate::catalog::inference::argument_error;
use runmat_types::{InferenceDiagnostic, NumericDomain, StorageFact, ValueFact, ValueKindFact};

pub(super) fn arguments(admitted: &AdmittedCall<'_>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(source) = admitted.source {
        real_numeric_or_logical(source, 0, "rescale input", diagnostics);
    }
    for (index, operand) in &admitted.operands {
        real_numeric_or_logical(operand, *index, "rescale bound", diagnostics);
    }
}

fn real_numeric_or_logical(
    value: &ValueFact,
    index: usize,
    label: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let valid = match value.kind {
        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
        ValueKindFact::Logical | ValueKindFact::Unknown => true,
        _ => false,
    };
    if !valid || matches!(value.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RESCALE-INPUT",
            format!("{label} must be a full real numeric or logical value"),
            index,
        ));
    }
}
