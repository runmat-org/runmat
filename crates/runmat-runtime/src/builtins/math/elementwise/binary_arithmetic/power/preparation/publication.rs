//! Publication precedes selected host confirmation. Both keep the same input
//! lease in the outer owner; failures never confirm a branch.
use super::*;
use crate::builtins::math::host_route::NumericHostRoute;
pub(super) struct Confirmation {
    pub(super) route: NumericHostRoute,
    pub(super) evidence: runmat_accelerate_api::NumericalDomainEvidence,
}
pub(super) fn publish<'a, P, Published>(
    runtime: Option<&'a RuntimeContext>,
    base: RootedValue,
    publish: P,
    confirmation: Option<Confirmation>,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    P: FnOnce(RootedValue) -> Published + 'a,
    Published: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    crate::execution::after_result(publish(base), move |output| async move {
        if let Some(Confirmation { route, evidence }) = confirmation {
            route.confirm_numerical_host_branch(
                runtime.expect("selected host route has its original runtime"),
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                evidence,
            );
        }
        Ok(output)
    })
}
