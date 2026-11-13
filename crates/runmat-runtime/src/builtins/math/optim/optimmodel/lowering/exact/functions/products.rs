//! Original exact product evaluation through the same factor geometry.
use super::super::super::factors::Factors;
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering::exact) async fn evaluate(
    input: Data,
    operation: Operation,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let mut plan = Factors::prepare(operation, &input.shape, arguments, &shape, policy, c).await?;
    let count = evaluation::element_count(&shape, policy)?;
    let one = Rational::from_usize(1);
    let mut values = data::buffer(count, policy)?;
    for _ in 0..count {
        values.push(Some(one.clone()));
        c.completed(1).await?;
    }
    match &mut plan {
        Factors::Reduction(reduction) => {
            for row in 0..input.count(policy)? {
                let target = reduction
                    .project(row)
                    .ok_or_else(|| policy.invalid("exact factor source exceeds reduction"))?;
                values[target] = match (values[target].take(), input.at(row, policy)?) {
                    (Some(a), Some(b)) => Some(a.mul(&b, policy, c).await?),
                    _ => None,
                };
                c.completed(1).await?;
            }
        }
        Factors::Scan(scan) => {
            let mut running = Some(one.clone());
            for step in scan.steps(false) {
                if step.first {
                    running = Some(one.clone());
                }
                running = match (running, input.at(step.index, policy)?) {
                    (Some(a), Some(b)) => Some(a.mul(&b, policy, c).await?),
                    _ => None,
                };
                values[step.index] = running.clone();
                c.completed(1).await?;
            }
        }
    }
    Data::explicit(shape, values, policy)
}
