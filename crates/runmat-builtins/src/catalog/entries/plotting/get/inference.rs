use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut handle_class_known = true;
    let mut known_control = false;
    if request.arguments.is_empty() || request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GET-ARITY",
            "get expects one handle and an optional property name",
            0,
        ));
    }
    if let Some(handle) = request.arguments.first() {
        if matches!(
            handle.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) || match &handle.kind {
            ValueKindFact::Unknown => {
                handle_class_known = false;
                false
            }
            ValueKindFact::Numeric(number) => {
                number.domain == NumericDomain::Complex || handle.storage == StorageFact::Sparse
            }
            ValueKindFact::Object(object) => {
                match (object.runtime_class.as_ref(), object.handle_semantics) {
                    (Some(identity), Some(true)) => {
                        known_control = standard::is_graphics_property_control(identity);
                        !known_control
                    }
                    (Some(_), Some(false)) => true,
                    _ => {
                        handle_class_known = false;
                        false
                    }
                }
            }
            _ => true,
        } {
            diagnostics.push(argument_error(
                "RM-CATALOG-GET-HANDLE",
                "graphics handles must be host scalar identities",
                0,
            ));
        }
        if handle.shape.element_count().is_some_and(|n| n != 1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GET-HANDLE",
                "graphics handle must be scalar",
                0,
            ));
        }
    }
    if let Some(property) = request.arguments.get(1) {
        if !matches!(
            property.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || matches!(
            property.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) || known_invalid_text_scalar(property)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GET-PROPERTY",
                "property name must be host text",
                1,
            ));
        }
    }
    let literal_property = request.literals.literal_string_at(1);
    if known_control
        && literal_property
            .as_ref()
            .is_some_and(|name| name.eq_ignore_ascii_case("Type"))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-GET-PROPERTY",
            "graphics control objects do not expose a Type property",
            1,
        ));
    }
    let mut output = if request.arguments.len() == 1 && handle_class_known {
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            Default::default(),
            false,
        )))
    } else if handle_class_known
        && literal_property.is_some_and(|name| {
            (!known_control && name.eq_ignore_ascii_case("Type"))
                || (known_control && name.eq_ignore_ascii_case("Enable"))
        })
    {
        ValueFact::scalar(ValueKindFact::String)
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    output.residency = ResidencyFact::Host;
    if !diagnostics.is_empty() {
        output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
