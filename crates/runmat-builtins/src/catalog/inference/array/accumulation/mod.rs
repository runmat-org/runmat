mod indexed;

#[cfg(test)]
mod tests;

use crate::{AccumulationInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: AccumulationInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        AccumulationInferenceRule::Indexed => indexed::infer(request, entry),
    }
}
