use crate::{BuiltinCatalogEntry, EnvironmentInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: EnvironmentInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        EnvironmentInferenceRule::Read => super::getenv::inference::infer(request, entry),
        EnvironmentInferenceRule::Exists => super::isenv::inference::infer(request, entry),
        EnvironmentInferenceRule::Set => super::setenv::inference::infer(request, entry),
        EnvironmentInferenceRule::Remove => super::unsetenv::inference::infer(request, entry),
    }
}
