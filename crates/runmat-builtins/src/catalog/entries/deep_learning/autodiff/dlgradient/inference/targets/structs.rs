//! Struct trees and the legacy scalar Learnables record.
use super::*;
use crate::DeepLearningParameterTableColumn as Column;
use runmat_types::{DynamicReason, StructFact};
use std::collections::BTreeMap;

fn legacy_record(fields: &BTreeMap<String, ValueFact>) -> bool {
    Column::ALL
        .iter()
        .all(|column| fields.contains_key(column.name()))
}

fn metadata_cannot_be_target(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Void
            | ValueKindFact::Symbolic
    )
}

fn gradient_fields(
    fields: &BTreeMap<String, ValueFact>,
    legacy_scalar_possible: bool,
) -> BTreeMap<String, ValueFact> {
    let special = legacy_scalar_possible && legacy_record(fields);
    let scalar_only_on_success = special
        && fields.iter().any(|(name, value)| {
            Column::from_name(name) != Some(Column::Value) && metadata_cannot_be_target(value)
        });
    fields
        .iter()
        .map(|(name, fact)| {
            let value = if special && Column::from_name(name) != Some(Column::Value) {
                if scalar_only_on_success {
                    // A scalar legacy record retains every metadata field,
                    // including fields beyond the three public table columns.
                    // The singleton StructArray route would try to
                    // differentiate an invalid metadata leaf and fail.
                    super::retained_metadata(fact)
                } else {
                    // Value::Struct preserves metadata; a singleton
                    // Value::StructArray maps every field. The fact model
                    // does not distinguish these two runtime variants.
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
            } else {
                super::gradient(fact)
            };
            (name.clone(), value)
        })
        .collect()
}

pub(super) fn gradient(target: &ValueFact, value: &StructFact) -> ValueFact {
    let may_be_scalar = target.shape.element_count() == Some(1);
    let fields = gradient_fields(&value.fields, may_be_scalar);
    let elements = value
        .elements
        .iter()
        .map(|element| gradient_fields(element, may_be_scalar && value.elements.len() == 1))
        .collect();
    let mut result = ValueFact::proven(
        ValueKindFact::Struct(StructFact {
            fields,
            fields_complete: value.fields_complete,
            elements,
            elements_complete: value.elements_complete,
        }),
        target.shape.clone(),
        target.storage,
    );
    result.alias = AliasFact::Unknown;
    result
}

fn check_fields(
    fields: &BTreeMap<String, ValueFact>,
    legacy_scalar_possible: bool,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if legacy_scalar_possible && legacy_record(fields) {
        // A scalar struct is admissible with opaque metadata. A singleton
        // struct array may fail here, but that possibility cannot justify a
        // diagnostic for a call that can succeed as a scalar record.
        super::check(&fields[Column::Value.name()], index, diagnostics);
    } else {
        for value in fields.values() {
            super::check(value, index, diagnostics);
        }
    }
}

pub(super) fn check(
    target: &ValueFact,
    value: &StructFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let may_be_scalar = target.shape.element_count() == Some(1);
    if value.elements_complete {
        for element in &value.elements {
            check_fields(
                element,
                may_be_scalar && value.elements.len() == 1,
                index,
                diagnostics,
            );
        }
    } else if target.shape.element_count().is_some_and(|count| count > 0) {
        check_fields(&value.fields, may_be_scalar, index, diagnostics);
        for element in &value.elements {
            check_fields(
                element,
                may_be_scalar && value.elements.len() == 1,
                index,
                diagnostics,
            );
        }
    }
}
