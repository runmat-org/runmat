use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 3 | 4) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ADDPOINTS-ARITY",
            "expected handle, X, Y, and optional Z",
            0,
        ));
    }
    if let Some(handle) = request.arguments.first() {
        if !matches!(
            handle.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || handle.shape.element_count().is_some_and(|count| count != 1)
            || matches!(handle.residency, ResidencyFact::Device { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ADDPOINTS-HANDLE",
                "expected a host scalar animated-line handle",
                0,
            ));
        }
    }
    let mut count = None;
    for (index, input) in request.arguments.iter().enumerate().skip(1) {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || input
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Complex)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ADDPOINTS-COORDINATE",
                "coordinates must be real numeric scalars or vectors",
                index,
            ));
        }
        if input.shape.known_dims().is_some_and(|dims| {
            dims.iter().skip(2).any(|d| d.is_some_and(|n| n != 1))
                || (dims.first().copied().flatten().is_some_and(|n| n != 1)
                    && dims.get(1).copied().flatten().is_some_and(|n| n != 1))
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ADDPOINTS-VECTOR",
                "coordinates must be scalars or vectors without non-singleton pages",
                index,
            ));
        }
        if let Some(elements) = input.shape.element_count() {
            if count.is_some_and(|prior| prior != elements) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ADDPOINTS-LENGTH",
                    "coordinate vectors must have equal element counts",
                    index,
                ));
            }
            count = Some(elements);
        }
    }
    finish_fixed_outputs(entry, request, vec![], diagnostics)
}
