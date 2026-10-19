mod next_exponent;
mod power;

use crate::{BuiltinCatalogEntry, PowerOfTwoInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: PowerOfTwoInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        PowerOfTwoInferenceRule::NextExponent => next_exponent::infer(request, entry),
        PowerOfTwoInferenceRule::Power => power::infer(request, entry),
    }
}
