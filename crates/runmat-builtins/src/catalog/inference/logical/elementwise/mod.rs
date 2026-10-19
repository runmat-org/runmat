mod binary;
mod output;
mod unary;

#[cfg(test)]
mod tests;

use crate::{BuiltinCatalogEntry, LogicalElementwiseRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: LogicalElementwiseRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        LogicalElementwiseRule::Binary(operator) => binary::infer(operator, request, entry),
        LogicalElementwiseRule::Unary(operator) => unary::infer(operator, request, entry),
    }
}
