mod call;
mod class;
mod shape;

#[cfg(test)]
mod tests;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    call::infer(request, entry)
}
