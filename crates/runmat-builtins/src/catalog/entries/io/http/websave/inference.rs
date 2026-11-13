use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-WEBSAVE-ARITY",
            "websave requires a filename and URL",
            0,
        ));
    }
    for index in 0..request.arguments.len().min(2) {
        if !possible_text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-WEBSAVE-TEXT",
                "filename and URL must be character rows or string scalars",
                index,
            ));
        }
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, diagnostics)
}

fn possible_text(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown)
        || super::super::super::repl_fs::text_input::scalar(value)
}
