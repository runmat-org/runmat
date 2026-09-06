use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs::directory_listing) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DIR-ARITY",
            "dir accepts at most two inputs",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        if !super::super::validation::scalar_text(argument) {
            diagnostics.push(argument_error(
                if index == 0 {
                    "RM-CATALOG-DIR-NAME"
                } else {
                    "RM-CATALOG-DIR-PATTERN"
                },
                if index == 0 {
                    "dir expects name to be a character row or string scalar"
                } else {
                    "dir expects pattern to be a character row or string scalar"
                },
                index,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        super::super::facts::metadata_listing(),
        diagnostics,
    )
}
