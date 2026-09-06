use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs::directory_lifecycle) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MKDIR-ARITY",
            "mkdir expects one folder or a parent and relative child",
            request.arguments.len().min(2),
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        if !super::super::validation::is_text_scalar(argument) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MKDIR-TEXT",
                "mkdir expects character-row or string-scalar folder names",
                index,
            ));
        }
    }
    finish_fixed_outputs(entry, request, super::super::facts::outputs(), diagnostics)
}
