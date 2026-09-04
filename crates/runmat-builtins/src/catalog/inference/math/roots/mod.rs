mod input;
mod literal;
mod principal;
mod real;

use crate::{BuiltinCatalogEntry, RootKind};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    kind: RootKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match kind {
        RootKind::Principal => principal::infer(request, entry),
        RootKind::RealOnly => real::infer(request, entry),
    }
}
