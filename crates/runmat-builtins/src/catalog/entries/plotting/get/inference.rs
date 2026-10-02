use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
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
        ) || !matches!(
            handle.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || handle
            .numeric()
            .is_some_and(|number| number.domain == NumericDomain::Complex)
            || handle.storage == StorageFact::Sparse
        {
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
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GET-PROPERTY",
                "property name must be host text",
                1,
            ));
        }
    }
    let mut output = if request.arguments.len() == 1 {
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            Default::default(),
            false,
        )))
    } else if request
        .literals
        .literal_string_at(1)
        .is_some_and(|name| name.eq_ignore_ascii_case("Type"))
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
