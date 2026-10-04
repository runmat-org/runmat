use std::collections::BTreeMap;

use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Hdf5Form {
    Display,
    Info,
    Read,
    Write,
    Attribute,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: Hdf5Form,
) -> CallInference {
    let count = request.arguments.len();
    let valid_arity = match form {
        Hdf5Form::Display => (1..=5).contains(&count),
        Hdf5Form::Info => matches!(count, 1 | 2),
        Hdf5Form::Read => matches!(count, 2 | 4 | 5),
        Hdf5Form::Write => matches!(count, 3 | 5 | 6),
        Hdf5Form::Attribute => count == 4,
    };
    let mut diagnostics = Vec::new();
    if !valid_arity {
        diagnostics.push(argument_error(
            "RM-CATALOG-HDF5-ARITY",
            "unsupported HDF5 argument count for this form",
            0,
        ));
    }
    for index in text_positions(form, count) {
        if let Some(value) = request.arguments.get(index) {
            if known_invalid_text(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-HDF5-TEXT",
                    "HDF5 filenames and object paths must be scalar text",
                    index,
                ));
            }
            if request
                .literals
                .literal_string_at(index)
                .is_some_and(|text| text.trim().is_empty())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-HDF5-EMPTY",
                    "HDF5 filenames and object paths must not be empty",
                    index,
                ));
            }
        }
    }
    if form == Hdf5Form::Display && valid_arity {
        validate_display_literals(request, &mut diagnostics);
    }
    if let Some(index) = data_position(form) {
        if let Some(value) = request.arguments.get(index) {
            if known_invalid_data(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-HDF5-DATA",
                    "HDF5 writes support dense real numeric, logical, string, and character data",
                    index,
                ));
            }
        }
    }
    let control_start = match form {
        Hdf5Form::Read => Some(2),
        Hdf5Form::Write => Some(3),
        _ => None,
    };
    if let Some(start) = control_start {
        for index in start..count {
            if request
                .arguments
                .get(index)
                .is_some_and(known_invalid_selection)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-HDF5-SELECTION",
                    "HDF5 selections must be real numeric vectors with positive integral entries",
                    index,
                ));
            }
        }
    }
    let outputs = match form {
        Hdf5Form::Display => vec![ValueFact::scalar(ValueKindFact::String)],
        Hdf5Form::Info => vec![metadata(request)],
        Hdf5Form::Read => vec![dataset_value()],
        Hdf5Form::Write | Hdf5Form::Attribute => Vec::new(),
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

fn text_positions(form: Hdf5Form, count: usize) -> Vec<usize> {
    match form {
        Hdf5Form::Display => (0..count).collect(),
        Hdf5Form::Info => (0..count.min(2)).collect(),
        Hdf5Form::Read | Hdf5Form::Write => (0..count.min(2)).collect(),
        Hdf5Form::Attribute => (0..count.min(3)).collect(),
    }
}

fn validate_display_literals(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let count = request.arguments.len();
    let token = |index| request.literals.literal_string_at(index);
    let mut require = |index, accepted: &[&str]| {
        if token(index).is_some_and(|value| {
            !accepted
                .iter()
                .any(|candidate| value.eq_ignore_ascii_case(candidate))
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-HDF5-DISPLAY",
                "h5disp mode must be 'simple' or 'min'; TextEncoding accepts only 'UTF-8'",
                index,
            ));
        }
    };
    match count {
        3 => {
            if token(1).is_some_and(|value| value.eq_ignore_ascii_case("TextEncoding")) {
                require(2, &["UTF-8"]);
            } else if token(1).is_none() {
                require(2, &["simple", "min", "UTF-8"]);
            } else {
                require(2, &["simple", "min"]);
            }
        }
        4 => {
            require(2, &["TextEncoding"]);
            require(3, &["UTF-8"]);
        }
        5 => {
            require(2, &["simple", "min"]);
            require(3, &["TextEncoding"]);
            require(4, &["UTF-8"]);
        }
        _ => {}
    }
}

fn data_position(form: Hdf5Form) -> Option<usize> {
    match form {
        Hdf5Form::Write => Some(2),
        Hdf5Form::Attribute => Some(3),
        _ => None,
    }
}

fn known_invalid_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .copied()
                .flatten()
                .is_some_and(|rows| rows != 1)
                || dims
                    .iter()
                    .skip(2)
                    .any(|extent| extent.is_some_and(|pages| pages != 1))
        }),
        _ => true,
    }
}

fn known_invalid_data(value: &ValueFact) -> bool {
    if matches!(value.storage, StorageFact::Sparse) {
        return true;
    }
    !matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical
            | ValueKindFact::String
            | ValueKindFact::Character
    )
}

fn known_invalid_selection(value: &ValueFact) -> bool {
    if matches!(value.storage, StorageFact::Sparse) {
        return true;
    }
    !matches!(
        value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    )
}

fn dataset_value() -> ValueFact {
    let mut value = ValueFact::unknown(DynamicReason::ExternalData);
    value.residency = ResidencyFact::Host;
    value
}

fn metadata(request: &CallRequest) -> ValueFact {
    let root = request.arguments.len() == 1
        || request.literals.literal_string_at(1).as_deref() == Some("/");
    let mut fields = BTreeMap::from([
        ("Name".into(), ValueFact::scalar(ValueKindFact::String)),
        (
            "Attributes".into(),
            ValueFact::unknown(DynamicReason::ExternalData),
        ),
    ]);
    if root {
        fields.insert("Filename".into(), ValueFact::scalar(ValueKindFact::String));
        for name in ["Groups", "Datasets", "Datatypes", "Links"] {
            fields.insert(name.into(), ValueFact::unknown(DynamicReason::ExternalData));
        }
    }
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, root)))
}
