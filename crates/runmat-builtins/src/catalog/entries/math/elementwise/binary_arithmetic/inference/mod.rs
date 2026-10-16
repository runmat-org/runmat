mod admission;
mod class;
mod diagnostics;
mod output;
mod policy;

#[cfg(test)]
mod admission_tests;

#[cfg(test)]
mod storage_tests;

#[cfg(test)]
pub(super) mod test_support;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use diagnostics::DiagnosticCollector;

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) use policy::{
    BinaryArithmeticInferencePolicy, MATCHING_SPARSE_REAL_RESULT, REAL_RESULT,
    RUNTIME_DEPENDENT_RESULT,
};

pub(super) fn infer(
    policy: BinaryArithmeticInferencePolicy,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let admitted = admission::inspect(request, true);
    output::infer(policy, admitted, request, entry)
}

/// The same canonical numeric fact before public CallInference wraps it in
/// effect/capability BTreeSets. Fusion's selected one-output preparation does
/// not consume those sets, and finite admission must not invent a bound for
/// their allocator-private node layout.
pub(super) fn infer_fusion_fact(
    policy: BinaryArithmeticInferencePolicy,
    request: &CallRequest,
) -> Result<runmat_types::ValueFact, Vec<runmat_types::InferenceDiagnostic>> {
    let admitted = admission::inspect(request, false);
    let (fact, diagnostics) = output::infer_fact(policy, admitted, request);
    if diagnostics.is_empty() {
        Ok(fact)
    } else {
        Err(Vec::new())
    }
}
