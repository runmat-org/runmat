//! Canonical unsigned little-endian limbs. No whole-number library operation.
use super::*;
pub(super) type Magnitude = Vec<u32>;
pub(super) async fn zeros(
    count: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    c.completed(0).await?;
    let mut out = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for _ in 0..count {
        out.push(0);
        c.completed(1).await?;
    }
    Ok(out)
}
pub(super) async fn copy(
    value: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    c.completed(0).await?;
    let mut out = runmat_value::exact_capacity_vec(value.len()).map_err(|e| policy.storage(e))?;
    for &word in value {
        out.push(word);
        c.completed(1).await?;
    }
    Ok(out)
}
pub(super) async fn trim(value: &mut Magnitude, c: &mut ScalarCooperation) -> BuiltinResult<()> {
    while value.last() == Some(&0) {
        value.pop();
        c.completed(1).await?;
    }
    Ok(())
}
pub(super) async fn compare(
    a: &[u32],
    b: &[u32],
    c: &mut ScalarCooperation,
) -> BuiltinResult<Ordering> {
    c.completed(0).await?;
    if a.len() != b.len() {
        return Ok(a.len().cmp(&b.len()));
    }
    for (&a, &b) in a.iter().zip(b).rev() {
        c.completed(1).await?;
        if a != b {
            return Ok(a.cmp(&b));
        }
    }
    Ok(Ordering::Equal)
}
pub(super) async fn add(
    a: &[u32],
    b: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    let count = a.len().max(b.len()).checked_add(1).ok_or_else(|| {
        policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
    })?;
    let mut out = zeros(count, policy, c).await?;
    let mut carry = 0u64;
    for (i, word) in out.iter_mut().enumerate() {
        let sum = u64::from(a.get(i).copied().unwrap_or(0))
            + u64::from(b.get(i).copied().unwrap_or(0))
            + carry;
        *word = sum as u32;
        carry = sum >> 32;
        c.completed(1).await?;
    }
    trim(&mut out, c).await?;
    Ok(out)
}
pub(super) async fn subtract(
    a: &[u32],
    b: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    let mut out = copy(a, policy, c).await?;
    subtract_in_place(&mut out, b, c).await?;
    Ok(out)
}
async fn subtract_in_place(
    a: &mut Magnitude,
    b: &[u32],
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let mut borrow = 0u64;
    for (i, word) in a.iter_mut().enumerate() {
        let rhs = u64::from(b.get(i).copied().unwrap_or(0)) + borrow;
        let lhs = u64::from(*word);
        *word = lhs.wrapping_sub(rhs) as u32;
        borrow = u64::from(lhs < rhs);
        c.completed(1).await?;
    }
    debug_assert_eq!(borrow, 0);
    trim(a, c).await
}
pub(super) async fn multiply(
    a: &[u32],
    b: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    if a.is_empty() || b.is_empty() {
        c.completed(0).await?;
        return Ok(Vec::new());
    }
    let count = a.len().checked_add(b.len()).ok_or_else(|| {
        policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
    })?;
    let mut out = zeros(count, policy, c).await?;
    for (i, &left) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &right) in b.iter().enumerate() {
            let sum = u64::from(left) * u64::from(right) + u64::from(out[i + j]) + carry;
            out[i + j] = sum as u32;
            carry = sum >> 32;
            c.completed(1).await?;
        }
        out[i + b.len()] = carry as u32;
        c.completed(1).await?;
    }
    trim(&mut out, c).await?;
    Ok(out)
}
#[path = "magnitude/division.rs"]
mod division;
#[path = "magnitude/gcd.rs"]
mod gcd_impl;
pub(super) use division::quotient;
pub(super) use gcd_impl::gcd;
