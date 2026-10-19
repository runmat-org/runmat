mod contract;
mod documentation;
mod inference;
mod mldivide;
mod mpower;
mod mrdivide;
mod mtimes;

use crate::{BuiltinCatalogEntry, MatrixArithmeticInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer_matrix_arithmetic(
    rule: MatrixArithmeticInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

pub use mldivide::*;
pub use mpower::*;
pub use mrdivide::*;
pub use mtimes::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    mldivide::mldivide_entry(),
    mpower::mpower_entry(),
    mrdivide::mrdivide_entry(),
    mtimes::mtimes_entry(),
];
