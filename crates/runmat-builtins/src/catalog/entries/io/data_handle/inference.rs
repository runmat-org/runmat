//! Statically provable receiver, argument, class, and output facts.

use std::collections::BTreeMap;

use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum DatasetForm {
    Path,
    Id,
    Version,
    Arrays,
    HasArray,
    Array,
    Attrs,
    GetAttr,
    SetAttr,
    SetAttrs,
    Begin,
    Snapshot,
    Refresh,
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn handle(class: StaticClassIdentity, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
}

pub(super) fn dataset() -> ValueFact {
    handle(
        super::super::data_family::DATASET_CLASS,
        BTreeMap::from([
            ("__data_path".into(), text()),
            ("__data_id".into(), text()),
            ("__data_version".into(), text()),
        ]),
    )
}

fn data_array() -> ValueFact {
    handle(
        super::super::data_family::DATA_ARRAY_CLASS,
        BTreeMap::from([
            ("__data_path".into(), text()),
            ("__array_name".into(), text()),
        ]),
    )
}

fn transaction() -> ValueFact {
    handle(
        super::super::data_family::DATA_TRANSACTION_CLASS,
        BTreeMap::from([("__data_path".into(), text()), ("__tx_id".into(), text())]),
    )
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

pub(super) fn dataset_infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: DatasetForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    let (minimum, maximum) = match form {
        DatasetForm::HasArray | DatasetForm::Array | DatasetForm::SetAttrs => (2, 2),
        DatasetForm::GetAttr => (2, 3),
        DatasetForm::SetAttr => (3, 3),
        DatasetForm::Snapshot => (2, usize::MAX),
        DatasetForm::Begin => (1, usize::MAX),
        _ => (1, 1),
    };
    if count < minimum || count > maximum {
        diagnostics.push(argument_error(
            "RM-DATA-HANDLE-ARITY",
            "Dataset method received an unsupported argument count",
            0,
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        let admissible = match &receiver.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                object.handle_semantics != Some(false)
                    && object
                        .runtime_class
                        .as_ref()
                        .is_none_or(|class| class.is(super::super::data_family::DATASET_CLASS))
            }
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-DATA-HANDLE-RECEIVER",
                "Dataset method requires a Dataset receiver",
                0,
            ));
        }
    }
    let text_argument = match form {
        DatasetForm::HasArray
        | DatasetForm::Array
        | DatasetForm::GetAttr
        | DatasetForm::SetAttr
        | DatasetForm::Snapshot => Some(1),
        _ => None,
    };
    if let Some(index) = text_argument {
        if request
            .arguments
            .get(index)
            .is_some_and(|arg| !text_possible(arg))
        {
            diagnostics.push(argument_error(
                "RM-DATA-HANDLE-TEXT",
                "Dataset name, key, or label must be scalar text",
                index,
            ));
        }
    }
    if matches!(form, DatasetForm::SetAttrs)
        && request.arguments.get(1).is_some_and(|arg| {
            !matches!(arg.kind, ValueKindFact::Struct(_) | ValueKindFact::Unknown)
        })
    {
        diagnostics.push(argument_error(
            "RM-DATA-HANDLE-ATTRS",
            "Dataset.set_attrs requires an attribute structure",
            1,
        ));
    }
    let output = match form {
        DatasetForm::Path | DatasetForm::Id | DatasetForm::Version | DatasetForm::Snapshot => {
            text()
        }
        DatasetForm::Arrays => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(text()),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Opaque,
        ),
        DatasetForm::HasArray | DatasetForm::SetAttr | DatasetForm::SetAttrs => {
            ValueFact::scalar(ValueKindFact::Logical)
        }
        DatasetForm::Array => data_array(),
        DatasetForm::Attrs => ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            BTreeMap::new(),
            false,
        ))),
        DatasetForm::GetAttr => ValueFact::unknown(DynamicReason::RuntimeValue),
        DatasetForm::Begin => transaction(),
        DatasetForm::Refresh => dataset(),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
