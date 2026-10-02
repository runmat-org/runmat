use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar},
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
            || match &handle.kind {
                ValueKindFact::Unknown => false,
                ValueKindFact::Numeric(number) => {
                    number.domain == NumericDomain::Complex || handle.storage == StorageFact::Sparse
                }
                ValueKindFact::Object(object) => {
                    match (object.runtime_class.as_ref(), object.handle_semantics) {
                        (Some(identity), Some(true)) => {
                            !standard::is_graphics_property_control(identity)
                        }
                        (Some(_), Some(false)) => true,
                        _ => false,
                    }
                }
                _ => true,
            }
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
        ) || known_invalid_text_scalar(name)
        {
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
