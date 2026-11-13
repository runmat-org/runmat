//! Deterministic adjusted-Halton, scaled-Householder OrthoMADS frames.
mod adjusted;
mod certificate;
mod halton;
#[cfg(test)]
mod tests;
use super::super::{buffer, invalid, mesh::Basis};
use super::level::Level;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
pub(super) use halton::Sequence;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

pub(in crate::builtins::math::optim::patternsearch) struct Householder {
    q: Vec<BigInt>,
    norm: BigInt,
    negative: Option<Vec<BigInt>>,
}
impl Householder {
    pub(super) async fn new(
        sequence: &mut Sequence,
        level: Level,
        basis: Basis,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let ray = sequence.direction(level.bits(), cooperation).await?;
        let q = adjusted::direction(&ray, level.bits(), cooperation).await?;
        drop(ray);
        let mut norm = BigInt::zero();
        let mut sum = BigInt::zero();
        for value in &q {
            norm += value * value;
            sum += value;
            cooperation.completed(1).await?;
        }
        let negative = if basis == Basis::Minimal {
            let mut values = buffer(q.len())?;
            for value in &q {
                let completed: BigInt = value * &sum * 2 - &norm;
                if !completed.to_f64().is_some_and(f64::is_finite) {
                    return Ok(None);
                }
                values.push(completed);
                cooperation.completed(1).await?;
            }
            Some(values)
        } else {
            None
        };
        let model = Self { q, norm, negative };
        if !certificate::admissible(&model, cooperation).await? {
            return Ok(None);
        }
        Ok(Some(model))
    }
    fn coefficient(&self, row: usize, column: usize) -> BigInt {
        let product: BigInt = &self.q[row] * &self.q[column] * 2;
        if row == column {
            &self.norm - product
        } else {
            -product
        }
    }
    pub fn count(&self) -> BuiltinResult<usize> {
        (if self.negative.is_some() {
            self.q.len().checked_add(1)
        } else {
            self.q.len().checked_mul(2)
        })
        .ok_or_else(|| invalid("OrthoMADS direction count overflow"))
    }
    pub fn bound(&self) -> Option<f64> {
        self.norm.to_f64().filter(|value| value.is_finite())
    }
    pub async fn direction(
        &self,
        index: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        if index >= self.count()? {
            return Err(invalid("OrthoMADS direction index out of range"));
        }
        let n = self.q.len();
        let mut ray = buffer(n)?;
        for row in 0..n {
            let value = if let Some(negative) = self.negative.as_ref().filter(|_| index == n) {
                negative[row].to_f64()
            } else {
                let value = self.coefficient(row, index % n);
                if index >= n { -value } else { value }.to_f64()
            };
            let Some(value) = value.filter(|value| value.is_finite()) else {
                return Ok(None);
            };
            ray.push(value);
            cooperation.completed(1).await?;
        }
        Ok(Some(ray))
    }
}
