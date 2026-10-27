//! Full-range exact proof arithmetic, not a numerical Value or solver.
//! Every variable-length pass uses the caller's original CPU cooperation.
use super::{model_policy::Policy, ScalarCooperation};
use crate::BuiltinResult;
use std::{cmp::Ordering, rc::Rc};
mod arithmetic;
mod export;
mod import;
mod magnitude;
#[cfg(test)]
mod tests;
use magnitude::Magnitude;
#[cfg(test)]
pub(in crate::builtins::math::optim) use tests::check_arithmetic;
#[derive(Clone)]
pub(in crate::builtins::math::optim) struct Rational {
    negative: bool,
    storage: Storage,
}
#[derive(Clone)]
enum Storage {
    Finite(f64),
    Integer(usize),
    Parts(Rc<Parts>),
}
struct Parts {
    numerator: Magnitude,
    denominator: Magnitude,
}
impl Rational {
    pub fn zero() -> Self {
        Self {
            negative: false,
            storage: Storage::Finite(0.),
        }
    }
    pub fn is_zero(&self) -> bool {
        match &self.storage {
            Storage::Finite(value) => *value == 0.,
            Storage::Integer(value) => *value == 0,
            Storage::Parts(parts) => parts.numerator.is_empty(),
        }
    }
    pub fn is_negative(&self) -> bool {
        self.negative && !self.is_zero()
    }
    pub fn is_positive(&self) -> bool {
        !self.negative && !self.is_zero()
    }
    pub fn negated(&self) -> Self {
        Self {
            negative: !self.negative && !self.is_zero(),
            storage: self.storage.clone(),
        }
    }
    pub fn abs(&self) -> Self {
        Self {
            negative: false,
            storage: self.storage.clone(),
        }
    }
    fn parts(negative: bool, numerator: Magnitude, denominator: Magnitude) -> Self {
        Self {
            negative: negative && !numerator.is_empty(),
            storage: Storage::Parts(Rc::new(Parts {
                numerator,
                denominator,
            })),
        }
    }
    async fn normalized(
        negative: bool,
        numerator: Magnitude,
        denominator: Magnitude,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if numerator.is_empty() {
            return Ok(Self::zero());
        }
        if denominator.is_empty() {
            return Err(policy.invalid("exact proof divisor is zero"));
        }
        let divisor = magnitude::gcd(&numerator, &denominator, policy, c).await?;
        let numerator = magnitude::quotient(&numerator, &divisor, policy, c).await?;
        let denominator = magnitude::quotient(&denominator, &divisor, policy, c).await?;
        Ok(Self::parts(negative, numerator, denominator))
    }
    pub async fn compare(
        &self,
        other: &Self,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Ordering> {
        c.completed(0).await?;
        if self.is_negative() != other.is_negative() {
            return Ok(self.is_negative().cmp(&other.is_negative()).reverse());
        }
        let a = self.materialize(policy, c).await?;
        let b = other.materialize(policy, c).await?;
        if Rc::ptr_eq(&a, &b) {
            return Ok(Ordering::Equal);
        }
        if magnitude::compare(&a.denominator, &b.denominator, c)
            .await?
            .is_eq()
        {
            let ordering = magnitude::compare(&a.numerator, &b.numerator, c).await?;
            return Ok(if self.negative {
                ordering.reverse()
            } else {
                ordering
            });
        }
        let left = magnitude::multiply(&a.numerator, &b.denominator, policy, c).await?;
        let right = magnitude::multiply(&b.numerator, &a.denominator, policy, c).await?;
        let result = magnitude::compare(&left, &right, c).await?;
        Ok(if self.negative {
            result.reverse()
        } else {
            result
        })
    }
}

impl std::ops::Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self {
        self.negated()
    }
}
impl std::ops::Neg for &Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        self.negated()
    }
}
