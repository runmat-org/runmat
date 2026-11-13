mod facts;
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
        facts::output(request.arguments.first()),
        validation::diagnostics(request),
    )
}
