use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count < 3 || count.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SET-PAIRS",
            "set expects a handle and one or more property/value pairs",
            count.saturating_sub(1),
        ));
    }
    if let Some(handle) = request.arguments.first() {
        if matches!(
            handle.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) || handle.shape.element_count().is_some_and(|n| n != 1)
            || !matches!(
                handle.kind,
                ValueKindFact::Unknown | ValueKindFact::Numeric(_)
            )
            || handle
                .numeric()
                .is_some_and(|number| number.domain == NumericDomain::Complex)
            || handle.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SET-HANDLE",
                "graphics handle must be a host scalar identity",
                0,
            ));
        }
    }
    for index in (1..count).step_by(2) {
        let name = &request.arguments[index];
        if !matches!(
            name.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || matches!(
            name.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SET-PROPERTY",
                "property names must be host text",
                index,
            ));
        }
    }
    let output = if diagnostics.is_empty() {
        ValueFact::scalar(ValueKindFact::String)
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
