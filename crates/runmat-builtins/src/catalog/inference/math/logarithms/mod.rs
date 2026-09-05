mod dissection;
mod literal_domain;
mod unary;

#[cfg(test)]
mod tests;

use crate::{BuiltinCatalogEntry, LogarithmKind};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    kind: LogarithmKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match kind {
        LogarithmKind::Binary => dissection::infer(request, entry),
        LogarithmKind::Natural | LogarithmKind::OnePlus | LogarithmKind::Common => {
            unary::infer(kind, request, entry)
        }
    }
}
