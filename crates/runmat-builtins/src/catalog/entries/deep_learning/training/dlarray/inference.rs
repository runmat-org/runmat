use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, ObjectFact, ResidencyFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-DL-DLARRAY-ARITY",
            "dlarray requires Data and at most one format argument",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(data) = request.arguments.first() {
        let valid = match &data.kind {
            ValueKindFact::Unknown | ValueKindFact::Logical => true,
            ValueKindFact::Numeric(number) => {
                matches!(number.class, NumericClass::Double | NumericClass::Single)
            }
            _ => false,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-DL-DLARRAY-DATA",
                "Data must be floating-point numeric or logical storage, not an integer or non-array object",
                0,
            ));
        }
    }
    if let Some(format) = request.arguments.get(1) {
        let valid = matches!(
            &format.residency,
            ResidencyFact::Host | ResidencyFact::Unknown
        ) && match &format.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::String => format.shape.element_count().is_none_or(|count| count == 1),
            ValueKindFact::Character => format
                .shape
                .known_dims()
                .is_none_or(|dims| dims.first().copied().flatten().is_none_or(|rows| rows == 1)),
            _ => false,
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-DL-DLARRAY-FORMAT",
                "format must be host text scalar or a row character vector",
                1,
            ));
        }
    }
    let data = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue));
    let mut result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([
                (crate::DLARRAY_DATA_PROPERTY.into(), data.clone()),
                ("Format".into(), ValueFact::scalar(ValueKindFact::String)),
                ("Labels".into(), ValueFact::scalar(ValueKindFact::String)),
            ]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        data.shape,
        StorageFact::Opaque,
    );
    result.residency = ResidencyFact::Host;
    finish_fixed(entry, request, result, diagnostics)
}
