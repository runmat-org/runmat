#[cfg(test)]
mod tests;
mod unary;

use crate::{BuiltinCatalogEntry, ExponentialKind};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    kind: ExponentialKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    unary::infer(kind, request, entry)
}
