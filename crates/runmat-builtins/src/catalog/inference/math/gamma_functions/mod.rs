mod common;
mod gamma;
mod gammaln;

use crate::{BuiltinCatalogEntry, GammaFunctionInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: GammaFunctionInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        GammaFunctionInferenceRule::Gamma => gamma::infer(request, entry),
        GammaFunctionInferenceRule::LogGamma => gammaln::infer(request, entry),
    }
}

#[cfg(test)]
mod tests;
