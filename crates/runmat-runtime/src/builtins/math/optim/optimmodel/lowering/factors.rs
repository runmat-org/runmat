//! Canonical product/scan routing shared by proposal and original exact proof.
use super::*;
use super::{linear::Rows, operations::Operation};
use crate::builtins::math::reduction::{cumsum, prod, sum};
pub(super) enum Factors {
    Reduction(sum::ReductionGeometry),
    Scan(cumsum::ScanGeometry),
}
impl Factors {
    pub(super) async fn prepare(
        operation: Operation,
        input: &[usize],
        arguments: &[Value],
        output: &[usize],
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        c.completed(0).await?;
        let runtime = c.runtime().cloned();
        crate::context::RuntimeScopeFactory::new(runtime, move || async move {
            let args = crate::call::FunctionArgumentView::from_slice(arguments);
            let plan = match operation {
                Operation::Product => {
                    Self::Reduction(prod::product_geometry(input, args, c.runtime()).await?)
                }
                Operation::Cumprod => Self::Scan(cumsum::scan_geometry(
                    runmat_builtins::CumulativeKind::Product,
                    input,
                    args,
                )?),
                _ => return Err(policy.invalid("invalid factor geometry operation")),
            };
            let actual = match &plan {
                Self::Reduction(plan) => plan.output_shape(),
                Self::Scan(_) => input,
            };
            if !crate::builtins::common::shape::array_shapes_equal(actual, output) {
                return Err(
                    policy.invalid("product geometry differs from original expression shape")
                );
            }
            Ok(plan)
        })
        .construct()
        .await
    }
    pub(super) fn outputs(&mut self, source: usize, policy: Policy) -> BuiltinResult<Rows> {
        Ok(match self {
            Self::Reduction(plan) => Rows::One(plan.project(source)),
            Self::Scan(plan) => Rows::Scan(
                plan.outputs(source)
                    .ok_or_else(|| policy.invalid("scan factor exceeds original geometry"))?,
            ),
        })
    }
}
