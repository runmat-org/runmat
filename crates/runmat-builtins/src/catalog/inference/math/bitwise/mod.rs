mod binary;
mod complement;
mod position;
mod shift;
mod swapbytes;

use crate::{BitwiseInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: BitwiseInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        BitwiseInferenceRule::Binary(operator) => binary::infer(operator, request, entry),
        BitwiseInferenceRule::Complement => complement::infer(request, entry),
        BitwiseInferenceRule::Get => position::infer_get(request, entry),
        BitwiseInferenceRule::Set => position::infer_set(request, entry),
        BitwiseInferenceRule::Shift => shift::infer(request, entry),
        BitwiseInferenceRule::SwapBytes => swapbytes::infer(request, entry),
    }
}

#[cfg(test)]
mod tests;
