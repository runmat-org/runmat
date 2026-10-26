//! Restoring binary division: each shift, compare and subtract cooperates.
use super::*;
pub(super) async fn divide(
    n: &[u32],
    d: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<(Magnitude, Magnitude)> {
    c.completed(0).await?;
    if d.is_empty() {
        return Err(policy.invalid("exact proof division by zero"));
    }
    if compare(n, d, c).await? == Ordering::Less {
        return Ok((Vec::new(), copy(n, policy, c).await?));
    }
    let bits = n
        .len()
        .checked_mul(32)
        .and_then(|v| v.checked_sub(n.last().copied().unwrap_or(0).leading_zeros() as usize))
        .ok_or_else(|| {
            policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
        })?;
    let mut q = zeros(n.len(), policy, c).await?;
    let capacity = d.len().checked_add(1).ok_or_else(|| {
        policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
    })?;
    let mut r = runmat_value::exact_capacity_vec(capacity).map_err(|e| policy.storage(e))?;
    for bit in (0..bits).rev() {
        let mut carry = (n[bit / 32] >> (bit % 32)) & 1;
        for word in &mut r {
            let next = *word >> 31;
            *word = (*word << 1) | carry;
            carry = next;
            c.completed(1).await?;
        }
        if carry != 0 {
            r.push(carry);
        }
        if compare(&r, d, c).await? != Ordering::Less {
            subtract_in_place(&mut r, d, c).await?;
            q[bit / 32] |= 1u32 << (bit % 32);
        }
        c.completed(1).await?;
    }
    trim(&mut q, c).await?;
    Ok((q, r))
}
pub(in crate::builtins::math::optim::common::proof_rational) async fn quotient(
    n: &[u32],
    d: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    if d == [1] {
        return copy(n, policy, c).await;
    }
    let (q, r) = divide(n, d, policy, c).await?;
    if !r.is_empty() {
        return Err(policy.invalid("exact proof normalization quotient has a remainder"));
    }
    Ok(q)
}
