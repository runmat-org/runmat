mod path;

use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let (output, diagnostics) = path::infer(request);
    finish_fixed(entry, request, output, diagnostics)
}
