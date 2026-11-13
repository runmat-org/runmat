use crate::{
    catalog::inference::{argument_error, finish_fixed, known_invalid_scalar_object_handle},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    let offset = request
        .arguments
        .first()
        .and_then(|first| match first.kind {
            ValueKindFact::String | ValueKindFact::Character => Some(0),
            ValueKindFact::Numeric(_) => {
                if first.shape.element_count().is_some_and(|n| n != 1)
                    || first
                        .numeric()
                        .is_some_and(|n| n.domain == NumericDomain::Complex)
                    || first.storage == StorageFact::Sparse
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-AXES-TARGET",
                        "axes target must be a scalar real numeric handle",
                        0,
                    ));
                }
                Some(1)
            }
            ValueKindFact::Unknown => None,
            ValueKindFact::Object(_) => {
                if known_invalid_scalar_object_handle(first, &[standard::FIGURE, standard::AXES]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-AXES-TARGET",
                        "axes target must be a scalar Figure or Axes handle",
                        0,
                    ));
                }
                Some(1)
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-AXES-INPUT",
                    "axes expects a figure/axes handle or property/value pairs",
                    0,
                ));
                None
            }
        });
    if let Some(offset) = offset {
        if !(request.arguments.len() - offset).is_multiple_of(2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-AXES-PAIRS",
                "axes properties require paired names and values",
                offset,
            ));
        }
        for index in (offset..request.arguments.len()).step_by(2) {
            if !matches!(
                request.arguments[index].kind,
                ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-AXES-NAME",
                    "axes property names must be text",
                    index,
                ));
            }
        }
    }
    // Supported values and aliases are interpreted by the shared graphics
    // property owner. A numeric target's value, not shape, selects create/select.
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
