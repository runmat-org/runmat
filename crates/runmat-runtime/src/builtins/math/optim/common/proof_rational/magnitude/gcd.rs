//! Binary GCD, with no indivisible variable-size remainder or normalization.
use super::*;
async fn trailing(
    value: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    for &word in value {
        c.completed(1).await?;
        if word != 0 {
            return count
                .checked_add(word.trailing_zeros() as usize)
                .ok_or_else(|| {
                    policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
                });
        }
        count = count.checked_add(32).ok_or_else(|| {
            policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
        })?;
    }
    Ok(count)
}
async fn right(value: &mut Magnitude, bits: usize, c: &mut ScalarCooperation) -> BuiltinResult<()> {
    let words = bits / 32;
    let shift = bits % 32;
    let len = value.len().saturating_sub(words);
    for i in 0..len {
        let low = value[i + words] >> shift;
        let high = if shift == 0 {
            0
        } else {
            value.get(i + words + 1).copied().unwrap_or(0) << (32 - shift)
        };
        value[i] = low | high;
        c.completed(1).await?;
    }
    value.truncate(len);
    trim(value, c).await
}
async fn left(
    value: &[u32],
    bits: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    let words = bits / 32;
    let shift = bits % 32;
    let count = value
        .len()
        .checked_add(words)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            policy.storage(runmat_value::ExactHostNumericAllocationError::LayoutOverflow)
        })?;
    let mut out = zeros(count, policy, c).await?;
    let mut carry = 0u64;
    for (i, &word) in value.iter().enumerate() {
        let next = (u64::from(word) << shift) | carry;
        out[i + words] = next as u32;
        carry = next >> 32;
        c.completed(1).await?;
    }
    out[value.len() + words] = carry as u32;
    trim(&mut out, c).await?;
    Ok(out)
}
pub(in crate::builtins::math::optim::common::proof_rational) async fn gcd(
    a: &[u32],
    b: &[u32],
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    c.completed(0).await?;
    if a == [1] || b == [1] {
        return copy(&[1], policy, c).await;
    }
    if a.is_empty() {
        return copy(b, policy, c).await;
    }
    if b.is_empty() {
        return copy(a, policy, c).await;
    }
    let mut a = copy(a, policy, c).await?;
    let mut b = copy(b, policy, c).await?;
    let az = trailing(&a, policy, c).await?;
    let bz = trailing(&b, policy, c).await?;
    let common = az.min(bz);
    right(&mut a, az, c).await?;
    right(&mut b, bz, c).await?;
    loop {
        match compare(&a, &b, c).await? {
            Ordering::Equal => break,
            Ordering::Greater => std::mem::swap(&mut a, &mut b),
            Ordering::Less => {}
        }
        subtract_in_place(&mut b, &a, c).await?;
        if b.is_empty() {
            break;
        }
        let zeros = trailing(&b, policy, c).await?;
        right(&mut b, zeros, c).await?;
    }
    left(&a, common, policy, c).await
}
