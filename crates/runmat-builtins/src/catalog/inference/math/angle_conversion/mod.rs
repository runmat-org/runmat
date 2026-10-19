use crate::{AngleConversionInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

mod unary;

pub(in crate::catalog::inference) fn infer(
    rule: AngleConversionInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    unary::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
