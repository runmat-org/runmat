use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-PARQUETINFO-ARITY",
            "exactly one filename is required",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|fact| !super::super::parquetread::scalar_text_input(fact))
        || request
            .literals
            .literal_string_at(0)
            .is_some_and(|name| name.is_empty())
    {
        diagnostics.push(argument_error(
            "RM-PARQUETINFO-FILENAME",
            "filename must be nonempty scalar text",
            0,
        ));
    }
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let text = ValueFact::scalar(ValueKindFact::String);
    let names = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let mut fields = BTreeMap::new();
    for name in [
        "FileSize",
        "NumRows",
        "NumRowGroups",
        "NumVariables",
        "Version",
    ] {
        fields.insert(name.into(), numeric.clone());
    }
    for name in ["Filename", "CreatedBy"] {
        fields.insert(name.into(), text.clone());
    }
    for name in ["VariableNames", "VariableTypes", "Compression", "Encoding"] {
        fields.insert(name.into(), names.clone());
    }
    fields.insert(
        "RowGroups".into(),
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            BTreeMap::new(),
            false,
        ))),
    );
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
        diagnostics,
    )
}
