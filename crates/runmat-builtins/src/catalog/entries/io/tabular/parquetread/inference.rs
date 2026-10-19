use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-PARQUETREAD-ARITY",
            "filename and complete name-value pairs are required",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|fact| !super::scalar_text_input(fact))
        || request
            .literals
            .literal_string_at(0)
            .is_some_and(|name| name.is_empty())
    {
        diagnostics.push(argument_error(
            "RM-PARQUETREAD-FILENAME",
            "filename must be nonempty scalar text",
            0,
        ));
    }
    let mut output_type = None;
    for index in (1..request.arguments.len()).step_by(2) {
        if !super::scalar_text_input(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-PARQUETREAD-OPTION-NAME",
                "option names must be scalar text",
                index,
            ));
            continue;
        }
        let Some(name) = request.literals.literal_string_at(index) else {
            // A later dynamic option may replace an earlier OutputType.
            output_type = None;
            continue;
        };
        match super::ParquetReadControl::parse(&name) {
            None => diagnostics.push(argument_error(
                "RM-PARQUETREAD-OPTION",
                "unsupported Parquet option",
                index,
            )),
            Some(super::ParquetReadControl::RowFilter | super::ParquetReadControl::RowTimes) => {
                diagnostics.push(argument_error(
                    "RM-PARQUETREAD-UNSUPPORTED",
                    "RowFilter and RowTimes are not implemented",
                    index,
                ))
            }
            Some(super::ParquetReadControl::OutputType) => {
                output_type = Some(index + 1);
                if request
                    .literals
                    .literal_string_at(index + 1)
                    .is_some_and(|kind| super::ParquetOutputType::parse_label(&kind).is_none())
                {
                    diagnostics.push(argument_error(
                        "RM-PARQUETREAD-OUTPUT-TYPE",
                        "only table output is implemented",
                        index + 1,
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some(index) = output_type {
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|kind| {
                super::ParquetOutputType::parse_label(&kind)
                    == Some(super::ParquetOutputType::Timetable)
            })
        {
            diagnostics.push(argument_error(
                "RM-PARQUETREAD-TIMETABLE",
                "timetable row-time synthesis is not implemented",
                index,
            ));
        }
    }
    // The scalar value is a table object, not a scalar struct or a matrix of
    // the external file's unknown row/column geometry.
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, output, diagnostics)
}
