use crate::{BuiltinCatalogEntry, DiscreteInferenceRule};
use runmat_types::{CallInference, CallRequest};

mod factorial;

pub(in crate::catalog::inference) fn infer(
    rule: DiscreteInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        DiscreteInferenceRule::Factorial => factorial::infer(request, entry),
    }
}
