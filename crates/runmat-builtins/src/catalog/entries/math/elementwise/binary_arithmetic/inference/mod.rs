mod admission;
mod class;
mod output;
mod policy;

#[cfg(test)]
pub(super) mod test_support;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) use policy::{
    BinaryArithmeticInferencePolicy, REAL_RESULT, RUNTIME_DEPENDENT_RESULT,
};

pub(super) fn infer(
    policy: BinaryArithmeticInferencePolicy,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let admitted = admission::inspect(request);
    output::infer(policy, admitted, request, entry)
}
