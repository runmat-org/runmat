mod input;
mod literal;
mod principal;
mod real;

use crate::{BuiltinCatalogEntry, RootKind};
use runmat_types::{CallInference, CallRequest, ValueFact};

pub(in crate::catalog::inference) fn infer_principal_fusion_fact(
    request: &CallRequest,
) -> Option<ValueFact> {
    principal::infer_fusion_fact(request)
}

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
