use super::{FilewriteEncoding, FilewriteMode, FilewriteOptionName};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-ARITY",
            "filewrite requires a filename and data",
            0,
        ));
        return result;
    }
    if invalid_text(&request.arguments[0]) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-PATH",
            "filename must be scalar text",
            0,
        ));
    }
    if invalid_data(&request.arguments[1]) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-DATA",
            "data must be text, logical, or real numeric storage",
            1,
        ));
    }
    // The first trailing text may be a positional encoding; subsequent
    // name/value pairs are validated at runtime, including duplicate names.
    for (index, value) in request.arguments.iter().enumerate().skip(2) {
        if invalid_text(value) {
            result.diagnostics.push(argument_error(
                "RM-CATALOG-FILEWRITE-OPTION-TEXT",
                "filewrite options must be scalar text",
                index,
            ));
        }
    }
    validate_options(request, &mut result.diagnostics);
    result
}

fn validate_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let tail = request.arguments.len().saturating_sub(2);
    if tail == 0 {
        return;
    }
    let first_literal = request.literals.literal_string_at(2);
    let first_name = first_literal
        .as_deref()
        .and_then(FilewriteOptionName::from_label);
    let positional = match first_literal.as_deref() {
        Some(_) => first_name.is_none(),
        None => tail % 2 == 1,
    };
    if positional
        && first_literal
            .as_deref()
            .is_some_and(|label| FilewriteEncoding::from_label(label).is_none())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-ENCODING",
            "unsupported positional encoding",
            2,
        ));
    }
    let start = if positional { 3 } else { 2 };
    if !(request.arguments.len() - start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILEWRITE-OPTION-ARITY",
            "filewrite options require name/value pairs",
            start,
        ));
        return;
    }
    let mut seen_encoding = positional;
    let mut seen_write_mode = false;
    for index in (start..request.arguments.len()).step_by(2) {
        let Some(name_literal) = request.literals.literal_string_at(index) else {
            continue;
        };
        let Some(name) = FilewriteOptionName::from_label(&name_literal) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILEWRITE-OPTION-NAME",
                "supported option names are Encoding and WriteMode",
                index,
            ));
            continue;
        };
        let duplicate = match name {
            FilewriteOptionName::Encoding => std::mem::replace(&mut seen_encoding, true),
            FilewriteOptionName::WriteMode => std::mem::replace(&mut seen_write_mode, true),
        };
        if duplicate {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILEWRITE-DUPLICATE-OPTION",
                "duplicate filewrite option",
                index,
            ));
        }
        if let Some(value) = request.literals.literal_string_at(index + 1) {
            let valid = match name {
                FilewriteOptionName::Encoding => FilewriteEncoding::from_label(&value).is_some(),
                FilewriteOptionName::WriteMode => FilewriteMode::from_label(&value).is_some(),
            };
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FILEWRITE-OPTION-VALUE",
                    "unsupported filewrite option value",
                    index + 1,
                ));
            }
        }
    }
}

fn invalid_text(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|dim| dim.is_some_and(|dim| dim != 1))
        }),
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

fn invalid_data(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String => false,
        ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Complex,
        ValueKindFact::Logical => false,
        _ => true,
    }
}
