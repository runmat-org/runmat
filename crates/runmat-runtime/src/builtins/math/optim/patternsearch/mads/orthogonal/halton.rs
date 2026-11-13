//! Exact deterministic Halton directions and the OrthoMADS index schedule.
use super::super::super::{buffer, invalid};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(in crate::builtins::math::optim::patternsearch::mads) struct Sequence {
    primes: Vec<usize>,
    deepest: Option<usize>,
    maximum_index: usize,
}
impl Sequence {
    pub async fn new(n: usize, cooperation: &mut ScalarCooperation) -> BuiltinResult<Self> {
        if n == 0 {
            return Err(invalid("OrthoMADS dimension must be positive"));
        }
        let mut primes = buffer(n)?;
        let mut candidate = 2usize;
        while primes.len() < n {
            let mut prime = true;
            for &divisor in &primes {
                if divisor > candidate / divisor {
                    break;
                }
                if candidate.is_multiple_of(divisor) {
                    prime = false;
                    break;
                }
                cooperation.completed(1).await?;
            }
            if prime {
                primes.push(candidate);
            }
            if primes.len() < n {
                candidate = candidate
                    .checked_add(if candidate == 2 { 1 } else { 2 })
                    .ok_or_else(|| invalid("OrthoMADS prime extent overflow"))?;
            }
            cooperation.completed(1).await?;
        }
        Ok(Self {
            primes,
            deepest: None,
            maximum_index: 0,
        })
    }

    pub async fn direction(
        &mut self,
        level: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<BigRational>> {
        let index = if self.deepest.is_none_or(|prior| level >= prior) {
            self.deepest = Some(level);
            level
                .checked_add(self.primes.len())
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| invalid("OrthoMADS Halton index overflow"))?
        } else {
            self.maximum_index
                .checked_add(1)
                .ok_or_else(|| invalid("OrthoMADS Halton index overflow"))?
        };
        self.maximum_index = self.maximum_index.max(index);
        let mut result = buffer(self.primes.len())?;
        for &base in &self.primes {
            let mut remaining = index;
            let mut numerator = BigInt::zero();
            let mut denominator = BigInt::one();
            while remaining != 0 {
                numerator = numerator * base + remaining % base;
                denominator *= base;
                remaining /= base;
                cooperation.completed(1).await?;
            }
            result.push(BigRational::new(numerator * 2 - &denominator, denominator));
            cooperation.completed(1).await?;
        }
        Ok(result)
    }
}
