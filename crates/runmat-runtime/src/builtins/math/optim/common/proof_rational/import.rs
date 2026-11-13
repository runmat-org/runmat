//! Exact source IEEE bits and borrowed library limbs, without library arithmetic.
use super::*;
impl Rational {
    /// Finite scalar metadata only; limb construction is delayed until needed.
    pub fn from_float(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            negative: value.is_sign_negative() && value != 0.,
            storage: Storage::Finite(value.abs()),
        })
    }
    pub fn from_usize(value: usize) -> Self {
        Self {
            negative: false,
            storage: Storage::Integer(value),
        }
    }

    pub(super) async fn materialize(
        &self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Rc<Parts>> {
        c.completed(0).await?;
        let value = match &self.storage {
            Storage::Finite(value) => *value,
            Storage::Integer(value) => {
                return Ok(Rc::new(Parts {
                    numerator: word(*value as u64, policy, c).await?,
                    denominator: magnitude::copy(&[1], policy, c).await?,
                }))
            }
            Storage::Parts(parts) => return Ok(Rc::clone(parts)),
        };
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let mut mantissa = bits & ((1u64 << 52) - 1);
        if exponent != 0 {
            mantissa |= 1u64 << 52;
        }
        if mantissa == 0 {
            return Ok(Rc::new(Parts {
                numerator: Vec::new(),
                denominator: magnitude::copy(&[1], policy, c).await?,
            }));
        }
        let zeros = mantissa.trailing_zeros();
        mantissa >>= zeros;
        let power = if exponent == 0 {
            -1074
        } else {
            exponent - 1023 - 52
        } + zeros as i32;
        let (numerator, denominator) = if power >= 0 {
            (
                shifted(word(mantissa, policy, c).await?, power as usize, policy, c).await?,
                magnitude::copy(&[1], policy, c).await?,
            )
        } else {
            (
                word(mantissa, policy, c).await?,
                shifted(
                    magnitude::copy(&[1], policy, c).await?,
                    (-power) as usize,
                    policy,
                    c,
                )
                .await?,
            )
        };
        Ok(Rc::new(Parts {
            numerator,
            denominator,
        }))
    }
    pub async fn import(
        value: &num_rational::BigRational,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        c.completed(0).await?;
        async fn limbs(
            value: &num_bigint::BigInt,
            policy: Policy,
            c: &mut ScalarCooperation,
        ) -> BuiltinResult<Magnitude> {
            let digits = value.iter_u32_digits();
            let mut out =
                runmat_value::exact_capacity_vec(digits.len()).map_err(|e| policy.storage(e))?;
            for digit in digits {
                out.push(digit);
                c.completed(1).await?;
            }
            Ok(out)
        }
        let numerator = limbs(value.numer(), policy, c).await?;
        let denominator = limbs(value.denom(), policy, c).await?;
        if denominator.is_empty() {
            return Ok(None);
        }
        let negative = (value.numer().sign() == num_bigint::Sign::Minus)
            != (value.denom().sign() == num_bigint::Sign::Minus);
        Ok(Some(
            Self::normalized(negative, numerator, denominator, policy, c).await?,
        ))
    }
}
async fn shifted(
    value: Magnitude,
    bits: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Magnitude> {
    let words = bits / 32;
    let shift = bits % 32;
    let count = value
        .len()
        .checked_add(words)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| policy.invalid("exact source bit extent overflow"))?;
    let mut out = magnitude::zeros(count, policy, c).await?;
    let mut carry = 0u64;
    for (i, word) in value.into_iter().enumerate() {
        let next = (u64::from(word) << shift) | carry;
        out[words + i] = next as u32;
        carry = next >> 32;
        c.completed(1).await?;
    }
    let last = out.len() - 1;
    out[last] = carry as u32;
    magnitude::trim(&mut out, c).await?;
    Ok(out)
}

async fn word(value: u64, policy: Policy, c: &mut ScalarCooperation) -> BuiltinResult<Magnitude> {
    c.completed(0).await?;
    let count = if value == 0 {
        0
    } else if value >> 32 == 0 {
        1
    } else {
        2
    };
    let mut out = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    if count > 0 {
        out.push(value as u32);
        c.completed(1).await?;
    }
    if count > 1 {
        out.push((value >> 32) as u32);
        c.completed(1).await?;
    }
    Ok(out)
}
