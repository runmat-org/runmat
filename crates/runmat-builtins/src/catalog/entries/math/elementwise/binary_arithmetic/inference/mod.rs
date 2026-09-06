mod admission;
mod class;
mod output;

use crate::{BinaryArithmeticInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    operation: BinaryArithmeticInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let admitted = admission::inspect(request);
    output::infer(operation, admitted, request, entry)
}

#[cfg(test)]
mod tests;
