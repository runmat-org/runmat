//! Useful, source-sound facts for path-level dataset calls.

use std::collections::BTreeMap;

use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum PathForm {
    Create,
    Open,
    Exists,
    Delete,
    Copy,
    Move,
    Import,
    Export,
    List,
    Inspect,
}

fn text_possible(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => !fact.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => match fact.shape.known_dims() {
            None => true,
            Some(dims) => {
                dims.first()
                    .is_none_or(|rows| rows.is_none_or(|rows| rows == 1))
                    && dims
                        .iter()
                        .skip(2)
                        .all(|page| page.is_none_or(|page| page == 1))
            }
        },
        _ => false,
    }
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn dataset() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::DATASET_CLASS.owned()),
            properties: BTreeMap::from([
                ("__data_path".into(), text()),
                ("__data_id".into(), text()),
                ("__data_version".into(), text()),
            ]),
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn path_list() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(text()),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Opaque,
    )
}

fn inspection() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([
            ("path".into(), text()),
            ("id".into(), text()),
            ("arrayCount".into(), double()),
            ("updatedAt".into(), text()),
        ]),
        true,
    )))
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: PathForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let expected = match form {
        PathForm::Create | PathForm::Copy | PathForm::Move => 2,
        PathForm::Import | PathForm::Export => 3,
        _ => 1,
    };
    if request.arguments.len() != expected {
        diagnostics.push(argument_error(
            "RM-DATA-PATH-ARITY",
            "data path operation received an unsupported argument count",
            0,
        ));
    }
    for index in 0..request.arguments.len() {
        if matches!(form, PathForm::Create) && index == 1 {
            match &request.arguments[index].kind {
                ValueKindFact::Struct(schema) => {
                    if schema.fields_complete && !schema.fields.contains_key("arrays") {
                        diagnostics.push(argument_error(
                            "RM-DATA-SCHEMA",
                            "data.create schema requires an arrays structure",
                            index,
                        ));
                    } else if schema.fields.get("arrays").is_some_and(|arrays| {
                        !matches!(
                            arrays.kind,
                            ValueKindFact::Struct(_) | ValueKindFact::Unknown
                        )
                    }) {
                        diagnostics.push(argument_error(
                            "RM-DATA-SCHEMA",
                            "data.create schema.arrays must be a structure",
                            index,
                        ));
                    }
                }
                ValueKindFact::Unknown => {}
                _ => diagnostics.push(argument_error(
                    "RM-DATA-SCHEMA",
                    "data.create schema must be a structure",
                    index,
                )),
            }
        } else if !text_possible(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-DATA-PATH-TEXT",
                "data path and format inputs must be scalar text",
                index,
            ));
        }
    }
    if matches!(form, PathForm::Import | PathForm::Export)
        && request
            .literals
            .literal_string_at(1)
            .is_some_and(|format| !format.eq_ignore_ascii_case("data"))
    {
        diagnostics.push(argument_error(
            "RM-DATA-FORMAT",
            "data import and export currently support only format='data'",
            1,
        ));
    }
    let output = match form {
        PathForm::Create | PathForm::Open | PathForm::Import => dataset(),
        PathForm::Exists
        | PathForm::Delete
        | PathForm::Copy
        | PathForm::Move
        | PathForm::Export => ValueFact::scalar(ValueKindFact::Logical),
        PathForm::List => path_list(),
        PathForm::Inspect => inspection(),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
