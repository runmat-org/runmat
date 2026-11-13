use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

pub(in crate::catalog::entries::io::repl_fs::path_syntax) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILEPARTS-ARITY",
            "fileparts expects exactly one filename",
            0,
        ));
    }
    let output = request.arguments.first().map(|input| {
        if super::super::validation::is_text_container(input) {
            super::super::facts::result_for_text(input)
        } else {
            diagnostics.push(argument_error("RM-CATALOG-FILEPARTS-TEXT", "fileparts expects a character row, string array, or cell array of character rows", 0));
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
    }).unwrap_or_else(|| ValueFact::unknown(DynamicReason::DynamicDispatch));
    finish_fixed_outputs(
        entry,
        request,
        vec![output.clone(), output.clone(), output],
        diagnostics,
    )
}
