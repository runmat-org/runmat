//! Optimization-model ancestry policy on the single generic registration owner.
use super::*;
use runmat_types::StaticClassIdentity;
const PARENTS: &[StaticClassIdentity] = &[
    standard::OPTIMIZATION_CONSTRAINT,
    standard::OPTIMIZATION_EXPRESSION,
];
pub(super) fn ensure(
    class: StaticClassIdentity,
    policy: Policy,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    super::super::common::native_class_registration::ensure(class, PARENTS, policy, runtime)
}
