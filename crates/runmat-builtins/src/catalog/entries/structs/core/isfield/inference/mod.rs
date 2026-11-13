mod output;
mod validation;

use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    finish_fixed(
        entry,
        request,
        output::fact(request.arguments.get(1)),
        validation::diagnostics(request),
    )
}
