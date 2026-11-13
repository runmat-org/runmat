mod call;
mod class;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    call::infer(request, entry)
}

#[cfg(test)]
mod tests;
