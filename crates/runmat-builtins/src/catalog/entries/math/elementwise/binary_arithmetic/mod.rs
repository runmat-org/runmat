mod contract;
mod documentation;
mod inference;
mod ldivide;
mod minus;
mod plus;
mod power;
mod rdivide;
mod times;

use crate::{BinaryArithmeticInferenceRule, BuiltinCatalogEntry};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer_binary_arithmetic(
    rule: BinaryArithmeticInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let policy = match rule {
        BinaryArithmeticInferenceRule::Add => plus::INFERENCE_POLICY,
        BinaryArithmeticInferenceRule::Subtract => minus::INFERENCE_POLICY,
        BinaryArithmeticInferenceRule::Multiply => times::INFERENCE_POLICY,
        BinaryArithmeticInferenceRule::RightDivide => rdivide::INFERENCE_POLICY,
        BinaryArithmeticInferenceRule::LeftDivide => ldivide::INFERENCE_POLICY,
        BinaryArithmeticInferenceRule::Power => power::INFERENCE_POLICY,
    };
    inference::infer(policy, request, entry)
}

pub use ldivide::*;
pub use minus::*;
pub use plus::*;
pub use power::*;
pub use rdivide::*;
pub use times::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    ldivide::ldivide_entry(),
    minus::minus_entry(),
    power::power_entry(),
    plus::plus_entry(),
    rdivide::rdivide_entry(),
    times::times_entry(),
];
