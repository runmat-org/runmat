//! A distinguished lattice direction is fixed for each refinement level.
use super::super::{buffer, invalid, push};
use super::{level::Level, sampling};
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::BuiltinResult;
use num_bigint::BigInt;
use std::rc::Rc;
pub(super) struct Direction {
    pub pivot: usize,
    pub values: Vec<BigInt>,
}
pub(in crate::builtins::math::optim::patternsearch) struct Refinements {
    dimension: usize,
    pub(super) orthogonal: Option<super::orthogonal::Sequence>,
    values: Vec<(Level, Rc<Direction>)>,
}
impl Refinements {
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn new(dimension: usize) -> BuiltinResult<Self> {
        if dimension == 0 {
            return Err(invalid("MADS dimension must be positive"));
        }
        Ok(Self {
            dimension,
            orthogonal: None,
            values: Vec::new(),
        })
    }
    pub(super) async fn direction(
        &mut self,
        level: Level,
        turn: &mut random::RandomTurn,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Rc<Direction>> {
        for (prior, value) in &self.values {
            if *prior == level {
                return Ok(value.clone());
            }
            cooperation.completed(1).await?;
        }
        let mut values = buffer(self.dimension)?;
        let pivot = sampling::pivot(turn, self.dimension, cooperation).await?;
        for coordinate in 0..self.dimension {
            values.push(if coordinate == pivot {
                sampling::diagonal(turn, level, cooperation).await?
            } else {
                sampling::off_diagonal(turn, level, cooperation).await?
            });
        }
        let value = Rc::new(Direction { pivot, values });
        push(&mut self.values, (level, value.clone()))?;
        Ok(value)
    }
}
