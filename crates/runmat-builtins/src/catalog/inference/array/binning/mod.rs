mod discretize;

#[cfg(test)]
mod tests;

use crate::{BinningInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: BinningInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        BinningInferenceRule::Discretize => discretize::infer(request, entry),
    }
}
