//! Static facts for managed transaction calls; runtime ownership is sealed separately.

use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

#[derive(Clone, Copy)]
pub(super) enum TransactionForm {
    Id,
    Status,
    Write,
    SetAttr,
    SetAttrs,
    Resize,
    Fill,
    DeleteArray,
    CreateArray,
    Commit,
    Abort,
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

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: TransactionForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    let (minimum, maximum) = match form {
        TransactionForm::Id | TransactionForm::Status | TransactionForm::Abort => (1, 1),
        TransactionForm::SetAttr | TransactionForm::CreateArray => (3, 3),
        TransactionForm::SetAttrs | TransactionForm::DeleteArray => (2, 2),
        TransactionForm::Write => (4, 4),
        TransactionForm::Resize => (3, 3),
        TransactionForm::Fill => (3, 4),
        TransactionForm::Commit => (1, 2),
    };
    if count < minimum || count > maximum {
        diagnostics.push(argument_error(
            "RM-DATA-TRANSACTION-ARITY",
            "DataTransaction method received an unsupported argument count",
            0,
        ));
    }
    if let Some(receiver) = request.arguments.first() {
        let admissible = match &receiver.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => {
                object.handle_semantics != Some(false)
                    && object.runtime_class.as_ref().is_none_or(|class| {
                        class.is(super::super::data_family::DATA_TRANSACTION_CLASS)
                    })
            }
            _ => false,
        };
        if !admissible {
            diagnostics.push(argument_error(
                "RM-DATA-TRANSACTION-RECEIVER",
                "DataTransaction method requires a managed DataTransaction receiver",
                0,
            ));
        }
    }
    let text_index = match form {
        TransactionForm::Write
        | TransactionForm::Resize
        | TransactionForm::Fill
        | TransactionForm::DeleteArray
        | TransactionForm::CreateArray
        | TransactionForm::SetAttr => Some(1),
        _ => None,
    };
    if let Some(index) = text_index {
        if request
            .arguments
            .get(index)
            .is_some_and(|fact| !text_possible(fact))
        {
            diagnostics.push(argument_error(
                "RM-DATA-TRANSACTION-TEXT",
                "Transaction array name or attribute key must be scalar text",
                index,
            ));
        }
    }
    let struct_index = match form {
        TransactionForm::SetAttrs => Some(1),
        TransactionForm::CreateArray => Some(2),
        TransactionForm::Commit if count == 2 => Some(1),
        _ => None,
    };
    if let Some(index) = struct_index {
        if request.arguments.get(index).is_some_and(|fact| {
            !matches!(fact.kind, ValueKindFact::Unknown | ValueKindFact::Struct(_))
                || fact.shape.is_definitely_nonscalar()
        }) {
            diagnostics.push(argument_error(
                "RM-DATA-TRANSACTION-STRUCT",
                "Transaction attributes, metadata, and commit options require a scalar structure",
                index,
            ));
        }
    }
    if matches!(form, TransactionForm::Resize)
        && request.arguments.get(2).is_some_and(|fact| {
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
            "RM-DATA-TRANSACTION-SHAPE",
            "Transaction resize requires real numeric shape dimensions",
            2,
        ));
    }
    let slice_index = match form {
        TransactionForm::Write => Some(2),
        TransactionForm::Fill if count > 3 => Some(3),
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
                "RM-DATA-TRANSACTION-SLICE",
                "Transaction slice must be a cell specification or ':' text",
                index,
            ));
        }
    }
    let output = match form {
        TransactionForm::Id | TransactionForm::Status => ValueFact::scalar(ValueKindFact::String),
        _ => ValueFact::scalar(ValueKindFact::Logical),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
