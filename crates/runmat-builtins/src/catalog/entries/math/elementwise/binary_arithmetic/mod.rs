mod contract;
mod inference;
mod plus;

use crate::{BinaryArithmeticInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer_binary_arithmetic(
    rule: BinaryArithmeticInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

pub use plus::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[plus::entry()];
