//! Exact value of an admitted original Euclidean norm-square pair.
use super::*;
pub(super) async fn evaluate(
    input: Data,
    exponent: Data,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    if exponent.count(policy)? != 1 || evaluation::element_count(&shape, policy)? != 1 {
        return Err(policy.invalid("atomic norm-square lost its original scalar geometry"));
    }
    let Some(exponent) = exponent.at(0, policy)? else {
        return Err(policy.invalid("atomic norm-square exponent is unassigned"));
    };
    if !exponent
        .compare(&Rational::from_usize(2), policy, c)
        .await?
        .is_eq()
    {
        return Err(policy.invalid("atomic norm-square lost its original exact exponent"));
    }
    let mut sum = Rational::zero();
    for row in 0..input.count(policy)? {
        let Some(value) = input.at(row, policy)? else {
            return Ok(Data::uniform(shape, None));
        };
        sum = sum
            .add(&value.mul(&value, policy, c).await?, policy, c)
            .await?;
        c.completed(1).await?;
    }
    Ok(Data::uniform(shape, Some(sum)))
}
