mod next_exponent;

#[cfg(test)]
mod tests;

use crate::{BuiltinCatalogEntry, PowerOfTwoInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: PowerOfTwoInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        PowerOfTwoInferenceRule::NextExponent => next_exponent::infer(request, entry),
    }
}
