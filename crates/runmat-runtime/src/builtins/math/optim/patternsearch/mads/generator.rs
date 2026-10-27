//! One frozen LTMADS triangular model; no random draw occurs at evaluation.
use super::super::{buffer, invalid, mesh::Basis, random::shuffle_in_turn};
use super::{
    certificate,
    level::Level,
    refinements::{Direction, Refinements},
    sampling,
};
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use std::rc::Rc;
pub(in crate::builtins::math::optim::patternsearch) struct Generator {
    pub(super) dimension: usize,
    pub(super) seed: Rc<Direction>,
    pub(super) rows: Vec<usize>,
    columns: Vec<usize>,
    pub(super) starts: Vec<usize>,
    pub(super) coefficients: Vec<BigInt>,
    pub(super) negative: Option<Vec<BigInt>>,
    basis: Basis,
}
pub(super) fn exact(value: &BigInt) -> bool {
    value
        .bits()
        .saturating_sub(value.trailing_zeros().unwrap_or(0))
        <= 53
}
impl Generator {
    pub(super) async fn new(
        refinements: &mut Refinements,
        level: Level,
        basis: Basis,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let n = refinements.dimension();
        let mut left = n - 1;
        let mut right = n;
        if left.is_multiple_of(2) {
            left /= 2;
        } else {
            right /= 2;
        }
        let count = left
            .checked_mul(right)
            .ok_or_else(|| invalid("MADS triangular extent overflow"))?;
        let mut coefficients = buffer(count)?;
        let mut starts = buffer(n - 1)?;
        let mut rows = buffer(n - 1)?;
        let mut columns = buffer(n)?;
        let mut negative = if basis == Basis::Minimal {
            Some(buffer(n)?)
        } else {
            None
        };
        for index in 0..n {
            columns.push(index);
            if let Some(sum) = &mut negative {
                sum.push(BigInt::zero());
            }
            cooperation.completed(1).await?;
        }
        let runtime = cooperation.runtime().cloned();
        let mut turn = random::acquire_original_turn(runtime.as_ref(), "patternsearch").await?;
        let seed = refinements.direction(level, &mut turn, cooperation).await?;
        for index in 0..n {
            if index != seed.pivot {
                rows.push(index);
            }
            cooperation.completed(1).await?;
        }
        for row in 0..n - 1 {
            starts.push(coefficients.len());
            for column in 0..=row {
                coefficients.push(if column == row {
                    sampling::diagonal(&mut turn, level, cooperation).await?
                } else {
                    sampling::off_diagonal(&mut turn, level, cooperation).await?
                });
            }
        }
        shuffle_in_turn(&mut turn, &mut rows, cooperation).await?;
        shuffle_in_turn(&mut turn, &mut columns, cooperation).await?;
        drop(turn);
        if let Some(sum) = &mut negative {
            for (value, source) in sum.iter_mut().zip(&seed.values) {
                *value -= source;
                cooperation.completed(1).await?;
            }
            for (row, &physical) in rows.iter().enumerate() {
                for column in 0..=row {
                    sum[physical] -= &coefficients[starts[row] + column];
                    cooperation.completed(1).await?;
                }
            }
        }
        let model = Self {
            dimension: n,
            seed,
            rows,
            columns,
            starts,
            coefficients,
            negative,
            basis,
        };
        if basis == Basis::Minimal {
            let mut all_exact = true;
            for value in model
                .seed
                .values
                .iter()
                .chain(&model.coefficients)
                .chain(model.negative.as_ref().expect("minimal completion"))
            {
                all_exact &= exact(value);
                cooperation.completed(1).await?;
            }
            if !all_exact && !certificate::positive(&model, cooperation).await? {
                return Ok(None);
            }
        }
        Ok(Some(model))
    }
    pub fn count(&self) -> BuiltinResult<usize> {
        match self.basis {
            Basis::Maximal => self.dimension.checked_mul(2),
            Basis::Minimal => self.dimension.checked_add(1),
        }
        .ok_or_else(|| invalid("MADS direction count overflow"))
    }
    pub fn bound(&self) -> Option<f64> {
        self.seed.values[self.seed.pivot]
            .to_f64()
            .filter(|value| value.is_finite())
            .map(f64::abs)
    }
    pub async fn direction(
        &self,
        index: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        if index >= self.count()? {
            return Err(invalid("MADS direction index out of range"));
        }
        let mut ray = buffer(self.dimension)?;
        if self.basis == Basis::Minimal && index == self.dimension {
            for value in self.negative.as_ref().expect("minimal completion") {
                let Some(value) = value.to_f64().filter(|value| value.is_finite()) else {
                    return Ok(None);
                };
                ray.push(value);
                cooperation.completed(1).await?;
            }
            return Ok(Some(ray));
        }
        for _ in 0..self.dimension {
            ray.push(0.0);
            cooperation.completed(1).await?;
        }
        let column = self.columns[index % self.dimension];
        if column == self.dimension - 1 {
            for (target, source) in ray.iter_mut().zip(&self.seed.values) {
                let Some(value) = source.to_f64().filter(|value| value.is_finite()) else {
                    return Ok(None);
                };
                *target = value;
                cooperation.completed(1).await?;
            }
        } else {
            for row in column..self.dimension - 1 {
                let Some(value) = self.coefficients[self.starts[row] + column]
                    .to_f64()
                    .filter(|value| value.is_finite())
                else {
                    return Ok(None);
                };
                ray[self.rows[row]] = value;
                cooperation.completed(1).await?;
            }
        }
        if index >= self.dimension {
            for value in &mut ray {
                *value = -*value;
                cooperation.completed(1).await?;
            }
        }
        Ok(Some(ray))
    }
}
