use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;

use super::{PrintControl, PRINT_ERROR_INVALID_OPTION};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-PRINT-ARITY",
            "print requires a filename",
            0,
        ));
    }
    let first_is_handle = request
        .arguments
        .first()
        .is_some_and(|first| matches!(first.kind, ValueKindFact::Numeric(_)));
    let start = usize::from(first_is_handle);
    if first_is_handle {
        let first = &request.arguments[0];
        if first.shape.element_count().is_some_and(|count| count != 1)
            || matches!(&first.kind, ValueKindFact::Numeric(number) if number.domain == NumericDomain::Complex)
            || first.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PRINT-FIGURE",
                "figure handle must be a real scalar",
                0,
            ));
        }
    }
    let mut filename_literals = 0usize;
    let mut devices = 0usize;
    for index in start..request.arguments.len() {
        let value = &request.arguments[index];
        if !matches!(
            value.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) || known_invalid_print_text(value)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PRINT-TEXT",
                "filename and options must be scalar text",
                index,
            ));
        }
        if let Some(text) = request.literals.literal_string_at(index) {
            let token = text.trim();
            if token.is_empty() || token.contains('\0') {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PRINT-TEXT",
                    "filename or option is empty or contains NUL",
                    index,
                ));
            } else if token.starts_with('-') {
                match PrintControl::decode(token) {
                    Ok(PrintControl::Device(_)) => devices += 1,
                    Ok(PrintControl::Resolution(_)) => {}
                    Err(_) => diagnostics.push(argument_error(
                        "RM-CATALOG-PRINT-OPTION",
                        PRINT_ERROR_INVALID_OPTION.message,
                        index,
                    )),
                }
            } else {
                filename_literals += 1;
            }
        }
    }
    if filename_literals > 1 || devices > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PRINT-FORM",
            "print accepts one filename and at most one device",
            start,
        ));
    }
    // An unknown scalar text operand may be the filename. Do not reject a
    // dynamic call merely because no literal filename is available.
    if filename_literals == 0 && request.arguments.len() == start {
        diagnostics.push(argument_error(
            "RM-CATALOG-PRINT-FILENAME",
            "print requires a filename",
            start,
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PRINT-OUTPUT",
            "print returns at most one RunMat status output",
            0,
        ));
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        diagnostics,
    )
}

fn known_invalid_print_text(value: &ValueFact) -> bool {
    if matches!(value.kind, ValueKindFact::Character) {
        // The historical print parser consumes every character in a 1-row
        // array, including pages. Do not reject a source-admitted form in
        // static inference while that compatibility behavior remains live.
        return matches!(&value.shape, ShapeFact::Shaped { dims }
            if dims.first().is_some_and(|dim| matches!(dim, DimensionFact::Known(rows) if *rows != 1)));
    }
    known_invalid_text_scalar(value)
}
