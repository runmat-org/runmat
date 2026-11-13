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

#[derive(Clone, Copy)]
pub(super) enum ArrayForm {
    Name,
    DType,
    Shape,
    Rank,
    ChunkShape,
    Codec,
    Read,
    Write,
    Resize,
    Fill,
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
        DatasetForm::Snapshot => (2, 2),
        DatasetForm::Begin => (1, 1),
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

pub(super) fn array_infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: ArrayForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    let (minimum, maximum) = match form {
        ArrayForm::Read => (1, 2),
        ArrayForm::Write => (2, 3),
        ArrayForm::Resize | ArrayForm::Fill => (2, 2),
        _ => (1, 1),
    };
    if count < minimum || count > maximum {
        diagnostics.push(argument_error(
            "RM-DATA-ARRAY-ARITY",
            "DataArray method received an unsupported argument count",
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
                        .is_none_or(|class| class.is(super::super::data_family::DATA_ARRAY_CLASS))
            }
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-DATA-ARRAY-RECEIVER",
                "DataArray method requires a managed DataArray receiver",
                0,
            ));
        }
    }
    let slice_index = match form {
        ArrayForm::Read if count >= 2 => Some(1),
        ArrayForm::Write if count >= 3 => Some(1),
        _ => None,
    };
    if let Some(index) = slice_index {
        if request.arguments.get(index).is_some_and(|fact| {
            !matches!(
                fact.kind,
                ValueKindFact::Unknown | ValueKindFact::Cell(_) | ValueKindFact::String
            )
        }) {
            diagnostics.push(argument_error(
                "RM-DATA-ARRAY-SLICE",
                "DataArray slice must be a cell specification or ':' text",
                index,
            ));
        }
    }
    if matches!(form, ArrayForm::Resize)
        && request.arguments.get(1).is_some_and(|fact| {
            !matches!(
                fact.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            )
        })
    {
        diagnostics.push(argument_error(
            "RM-DATA-ARRAY-SHAPE",
            "DataArray.resize requires real numeric shape dimensions",
            1,
        ));
    }
    let value_index = match form {
        ArrayForm::Write if count >= 3 => Some(2),
        ArrayForm::Write | ArrayForm::Fill => Some(1),
        _ => None,
    };
    if let Some(index) = value_index {
        if request.arguments.get(index).is_some_and(|fact| {
            !matches!(
                fact.kind,
                ValueKindFact::Unknown | ValueKindFact::Numeric(_)
            )
        }) {
            diagnostics.push(argument_error(
                "RM-DATA-ARRAY-VALUE",
                "DataArray payload must be numeric",
                index,
            ));
        }
    }
    let output = match form {
        ArrayForm::Name | ArrayForm::DType | ArrayForm::Codec => text(),
        ArrayForm::Shape | ArrayForm::ChunkShape => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
        ArrayForm::Rank => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        ArrayForm::Read => {
            let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
            fact.residency = ResidencyFact::Host;
            fact.storage = StorageFact::Dense;
            fact
        }
        ArrayForm::Write | ArrayForm::Resize | ArrayForm::Fill => {
            ValueFact::scalar(ValueKindFact::Logical)
        }
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
