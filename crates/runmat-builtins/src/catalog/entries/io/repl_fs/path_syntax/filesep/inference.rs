use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};
pub(in crate::catalog::entries::io::repl_fs::path_syntax) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = (!request.arguments.is_empty())
        .then(|| argument_error("RM-CATALOG-FILESEP-ARITY", "filesep accepts no inputs", 0))
        .into_iter()
        .collect();
    finish_fixed(
        entry,
        request,
        super::super::facts::character_scalar(),
        diagnostics,
    )
}
