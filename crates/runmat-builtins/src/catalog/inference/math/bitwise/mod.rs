mod swapbytes;

use crate::{BitwiseInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: BitwiseInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        BitwiseInferenceRule::SwapBytes => swapbytes::infer(request, entry),
    }
}

#[cfg(test)]
mod tests;
