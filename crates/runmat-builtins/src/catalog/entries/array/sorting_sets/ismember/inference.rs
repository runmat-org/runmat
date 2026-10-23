use super::super::{facts::*, set_facts::RecordMode};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    count(request, 2, &mut diagnostics);
    let a = argument(request, 0);
    let b = argument(request, 1);
    accepted_data(&a, true, false, &mut diagnostics, 0);
    accepted_data(&b, true, false, &mut diagnostics, 1);
    let mut rows = RecordMode::Elements;
    for index in 2..request.arguments.len() {
        match text_option(request, index, &mut diagnostics).as_deref() {
            Some("rows") => rows = RecordMode::Rows,
            Some(_) => diagnostics.push(argument_error(
                "RM-ISMEMBER-OPTION",
                "only the rows flag is supported",
                index,
            )),
            None if rows != RecordMode::Rows => rows = RecordMode::Unknown,
            None => {}
        }
    }
    let shape = match rows {
        RecordMode::Rows => {
            row_compatibility(&a, &b, &mut diagnostics);
            column(dimension(&a.shape, 0))
        }
        RecordMode::Elements => a.shape.clone(),
        RecordMode::Unknown => ShapeFact::Unknown,
    };
    let residency = first_resident(&a, &b);
    finish_fixed_outputs(
        entry,
        request,
        vec![
            logical(shape.clone(), residency.clone()),
            index(shape, residency),
        ],
        diagnostics,
    )
}
