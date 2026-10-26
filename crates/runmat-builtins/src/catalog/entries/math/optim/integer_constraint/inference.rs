use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry, IntegerConstraintField,
};
use runmat_types::{
    CallInference, CallRequest, ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGER-CONSTRAINT-ARITY",
            "expected property name/value pairs",
            request.arguments.len(),
        ));
    }
    let mut seen = Vec::new();
    for index in (0..request.arguments.len()).step_by(2) {
        let name = &request.arguments[index];
        if !matches!(
            name.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-INTEGER-CONSTRAINT-NAME",
                "property names must be text scalars",
                index,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(index) {
            match IntegerConstraintField::parse(&name) {
                Some(key) if !seen.contains(&key) => seen.push(key),
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-INTEGER-CONSTRAINT-PROPERTY",
                    "unknown or repeated property name",
                    index,
                )),
            }
        }
        if let Some(value) = request.arguments.get(index + 1) {
            let numeric = matches!(value.kind, ValueKindFact::Unknown)
                || matches!(
                    value.kind,
                    ValueKindFact::Numeric(runmat_types::NumericFact {
                        class: runmat_types::NumericClass::Double,
                        domain: runmat_types::NumericDomain::Real,
                    })
                );
            let invalid_shape = value.shape.element_count() != Some(0)
                && (value.shape.rank().is_some_and(|rank| rank > 2)
                    || value.shape.known_dims().is_some_and(|dims| {
                        dims.len() == 2
                            && dims[0].is_some_and(|n| n > 1)
                            && dims[1].is_some_and(|n| n > 1)
                    }));
            if !numeric || invalid_shape || value.storage == StorageFact::Sparse {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INTEGER-CONSTRAINT-INDICES",
                    "indices must be full real double vectors",
                    index + 1,
                ));
            }
        }
    }
    // Public COW edits can replace property values. Exact class/value identity
    // does not prove a persistent Double type for any mutable property.
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::INTEGER_CONSTRAINT.owned()),
            properties: std::collections::BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}
