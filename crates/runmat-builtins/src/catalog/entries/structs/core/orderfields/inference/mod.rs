mod output;
mod validation;

use crate::catalog::inference::finish_fixed_outputs;
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    finish_fixed_outputs(
        entry,
        request,
        output::facts(request.arguments.first()),
        validation::diagnostics(request),
    )
}
