use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;

use super::DrawNowControl;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DRAWNOW-ARITY",
            "drawnow accepts at most two control tokens",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let invalid = match argument.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Character => matches!(&argument.shape, ShapeFact::Shaped { dims }
                if dims.first().is_some_and(|dim| matches!(dim, DimensionFact::Known(rows) if *rows != 1))),
            ValueKindFact::String => known_invalid_text_scalar(argument),
            _ => true,
        };
        if invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-DRAWNOW-CONTROL",
                "drawnow controls must be scalar text",
                index,
            ));
        }
    }
    let literals: Option<Vec<String>> = (0..request.arguments.len())
        .map(|index| request.literals.literal_string_at(index))
        .collect();
    if let Some(literals) = literals {
        let tokens: Vec<&str> = literals.iter().map(String::as_str).collect();
        if DrawNowControl::decode(&tokens).is_err() {
            diagnostics.push(argument_error(
                "RM-CATALOG-DRAWNOW-CONTROL",
                "drawnow requires limitrate, nocallbacks, update, or expose in a supported form",
                0,
            ));
        }
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-DRAWNOW-OUTPUT",
            "drawnow returns at most one RunMat status output",
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
