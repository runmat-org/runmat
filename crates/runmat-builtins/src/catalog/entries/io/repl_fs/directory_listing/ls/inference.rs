use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io::repl_fs::directory_listing) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LS-ARITY",
            "ls accepts at most one input",
            1,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|argument| !super::super::super::text_input::scalar(argument))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-LS-NAME",
            "ls expects name to be a character row or string scalar",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        super::super::facts::name_listing(),
        diagnostics,
    )
}
