//! Exact normalization shared by original forward/reverse and quadratic algebra.
use super::*;
use crate::builtins::math::optim::common::proof_rational::Rational;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn denominators(
    plan: &mut Linear,
    inputs: usize,
    outputs: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Vec<Rational>> {
    let Linear::Mean(mean) = plan else {
        return Ok(Vec::new());
    };
    let mut values = runmat_value::exact_capacity_vec(outputs).map_err(|e| policy.storage(e))?;
    for _ in 0..outputs {
        values.push(Rational::zero());
        c.completed(1).await?;
    }
    if !mean.is_weighted() {
        let count = Rational::from_usize(mean.reduction.factor());
        for value in &mut values {
            *value = count.clone();
            c.completed(1).await?;
        }
        return Ok(values);
    }
    for source in 0..inputs {
        let row = mean
            .reduction
            .project(source)
            .ok_or_else(|| policy.invalid("exact mean source exceeds admitted projection"))?;
        let weight = Rational::from_float(mean.weight(source))
            .ok_or_else(|| policy.invalid("exact mean weight must be finite"))?;
        values[row] = values[row].add(&weight, policy, c).await?;
        c.completed(1).await?;
    }
    Ok(values)
}
