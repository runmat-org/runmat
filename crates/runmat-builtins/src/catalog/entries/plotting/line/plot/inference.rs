use crate::{
    catalog::inference::{argument_error, finish_fixed, known_invalid_scalar_object_handle},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-PLOT-DATA",
            "plot requires coordinate data",
            0,
        ));
    }
    if let Some(first) = request.arguments.first() {
        if matches!(first.kind, ValueKindFact::Object(_))
            && known_invalid_scalar_object_handle(first, &[standard::AXES])
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PLOT-TARGET",
                "plot object target must be a scalar Axes handle",
                0,
            ));
        }
    }
    // Values select an optional leading axes alias and series/options. Do not
    // infer that a scalar numeric first argument is necessarily an axes handle.
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
