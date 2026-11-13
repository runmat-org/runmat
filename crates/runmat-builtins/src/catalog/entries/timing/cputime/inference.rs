use crate::catalog::inference::{default_double_scalar, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, InferenceDiagnostic};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![InferenceDiagnostic::error(
            "RM-CATALOG-CPUTIME-ARITY",
            "cputime takes no inputs",
        )]
    };
    finish_fixed(entry, request, default_double_scalar(), diagnostics)
}
