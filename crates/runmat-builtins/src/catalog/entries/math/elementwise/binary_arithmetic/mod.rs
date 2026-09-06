mod contract;
mod inference;
mod minus;
mod plus;
mod times;

use crate::{BinaryArithmeticInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer_binary_arithmetic(
    rule: BinaryArithmeticInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

pub use minus::*;
pub use plus::*;
pub use times::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    minus::minus_entry(),
    plus::plus_entry(),
    times::times_entry(),
];
