//! Zero-safe original product rule over retained candidate operand values.
//! The existing affine constant-factor path remains allocation/behavior unchanged.
use super::*;
struct Summary {
    nonzero: Rational,
    zeros: usize,
}
impl Summary {
    fn one() -> Self {
        Self {
            nonzero: Rational::from_usize(1),
            zeros: 0,
        }
    }
    async fn consume(
        &mut self,
        value: &Rational,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if value.is_zero() {
            self.zeros = self
                .zeros
                .checked_add(1)
                .ok_or_else(|| policy.invalid("exact product zero count overflows usize"))?;
        } else {
            self.nonzero = self.nonzero.mul(value, policy, c).await?;
        }
        c.completed(1).await
    }
    async fn slope(
        &self,
        value: &Rational,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Rational> {
        if self.zeros == 0 {
            self.nonzero.div(value, policy, c).await
        } else if self.zeros == 1 && value.is_zero() {
            Ok(self.nonzero.clone())
        } else {
            Ok(Rational::zero())
        }
    }
}
fn operand(
    values: &Evaluated,
    root: GcHandle,
    row: usize,
    policy: Policy,
) -> BuiltinResult<Rational> {
    values
        .value(root, row, policy)?
        .ok_or_else(|| policy.invalid("original product operand is unassigned"))
}
pub(super) async fn propagate(
    state: &mut State<'_>,
    values: &Evaluated,
    operation: Operation,
    source: GcHandle,
    arguments: &[Value],
    shape: &[usize],
    weights: Weights,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if !state.active(source)? {
        return Ok(());
    }
    let policy = state.policy;
    let input = state
        .program
        .source
        .nodes
        .get(&source)
        .ok_or_else(|| policy.invalid("retained product lost original input"))?
        .node()
        .shape
        .clone();
    let mut plan = Factors::prepare(operation, &input, arguments, shape, policy, c).await?;
    let inputs = evaluation::element_count(&input, policy)?;
    match &mut plan {
        Factors::Reduction(reduction) => {
            let outputs = evaluation::element_count(shape, policy)?;
            let mut summaries =
                runmat_value::exact_capacity_vec(outputs).map_err(|e| policy.storage(e))?;
            for _ in 0..outputs {
                summaries.push(Summary::one());
                c.completed(1).await?;
            }
            for row in 0..inputs {
                let target = reduction
                    .project(row)
                    .ok_or_else(|| policy.invalid("retained factor exceeds reduction"))?;
                summaries[target]
                    .consume(&operand(values, source, row, policy)?, policy, c)
                    .await?;
            }
            for row in 0..inputs {
                let target = reduction
                    .project(row)
                    .ok_or_else(|| policy.invalid("retained factor exceeds reduction"))?;
                if let Some(weight) = weights.get(&target) {
                    if state.active_at(source, row)? {
                        let slope = summaries[target]
                            .slope(&operand(values, source, row, policy)?, policy, c)
                            .await?;
                        state
                            .add(source, row, weight.mul(&slope, policy, c).await?, c)
                            .await?;
                    }
                }
                c.completed(1).await?;
            }
        }
        Factors::Scan(scan) => {
            let mut before =
                runmat_value::exact_capacity_vec(inputs).map_err(|e| policy.storage(e))?;
            for _ in 0..inputs {
                before.push(Rational::zero());
                c.completed(1).await?;
            }
            let mut product = Rational::from_usize(1);
            for step in scan.steps(false) {
                if step.first {
                    product = Rational::from_usize(1);
                }
                before[step.index] = product.clone();
                product = product
                    .mul(&operand(values, source, step.index, policy)?, policy, c)
                    .await?;
                c.completed(1).await?;
            }
            let mut adjoint = Rational::zero();
            for step in scan.steps(true) {
                if step.first {
                    adjoint = Rational::zero();
                }
                if let Some(weight) = weights.get(&step.index) {
                    adjoint = adjoint.add(weight, policy, c).await?;
                }
                if state.active_at(source, step.index)? {
                    state
                        .add(
                            source,
                            step.index,
                            adjoint.mul(&before[step.index], policy, c).await?,
                            c,
                        )
                        .await?;
                }
                adjoint = adjoint
                    .mul(&operand(values, source, step.index, policy)?, policy, c)
                    .await?;
                c.completed(1).await?;
            }
        }
    }
    Ok(())
}
