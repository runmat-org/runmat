use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-SURF-DATA",
            "surf requires Z or X,Y,Z surface coordinates",
            0,
        ));
    }
    // Positional color data and name/value parsing remain the runtime's typed
    // plotting boundary. A numeric second argument is C for the two-data form,
    // but Y for the three-data form, so no speculative shape inference here.
    if let Some(first) = request.arguments.first() {
        if !matches!(
            first.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SURF-NUMERIC",
                "surf coordinate data must be numeric",
                0,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}
