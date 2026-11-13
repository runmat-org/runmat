//! Quasirandom construction and the documented high-dimensional Latin path.
//! Random draws use one short original-stream turn, retired before callbacks.
use super::{invalid, NAME};
use crate::builtins::common::random;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use std::collections::VecDeque;

pub(super) enum Sampler {
    Halton {
        bases: Vec<usize>,
        shift: Vec<f64>,
        next: Option<usize>,
    },
    Latin {
        remaining: VecDeque<Vec<f64>>,
    },
}
fn capacity<T>(n: usize) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(n).map_err(|source| {
        crate::build_runtime_error("surrogateopt: construction workspace allocation failed")
            .with_builtin(NAME)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    })
}
async fn uniforms(
    turn: &mut random::RandomTurn,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let (runtime, budget) = cooperation.runtime_budget();
    match budget {
        Some(budget) => random::generate_uniform_cooperative(turn, n, NAME, runtime, budget).await,
        None => random::generate_uniform(turn, n, NAME),
    }
}
impl Sampler {
    pub(super) async fn new(
        problem_variables: usize,
        dimensions: usize,
        construction_rows: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if dimensions == 0 || dimensions > problem_variables || construction_rows == 0 {
            return Err(invalid("invalid construction sequence geometry"));
        }
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            NAME,
            "finite host-memory execution has no complete construction resource bound",
        )?;
        if problem_variables <= 500 {
            let mut bases = capacity(dimensions)?;
            let mut candidate = 2usize;
            while bases.len() < dimensions {
                let mut prime = true;
                for &divisor in &bases {
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
                    bases.push(candidate);
                }
                candidate = candidate
                    .checked_add(1)
                    .ok_or_else(|| invalid("quasirandom prime extent overflow"))?;
                cooperation.completed(1).await?;
            }
            let mut turn = random::acquire_original_turn(cooperation.runtime(), NAME).await?;
            let shift = uniforms(&mut turn, dimensions, cooperation).await?;
            drop(turn);
            return Ok(Self::Halton {
                bases,
                shift,
                next: Some(1),
            });
        }
        construction_rows
            .checked_mul(dimensions)
            .ok_or_else(|| invalid("Latin construction extent overflow"))?;
        let mut remaining = VecDeque::new();
        remaining
            .try_reserve_exact(construction_rows)
            .map_err(|source| {
                crate::build_runtime_error("surrogateopt: Latin row allocation failed")
                    .with_builtin(NAME)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(source)
                    .build()
            })?;
        for _ in 0..construction_rows {
            let mut row = capacity(dimensions)?;
            row.resize(dimensions, 0.0);
            remaining.push_back(row);
            cooperation.completed(1).await?;
        }
        let mut permutation = capacity(construction_rows)?;
        let mut turn = random::acquire_original_turn(cooperation.runtime(), NAME).await?;
        for column in 0..dimensions {
            permutation.clear();
            for row in 0..construction_rows {
                permutation.push(row);
                cooperation.completed(1).await?;
            }
            let draws =
                uniforms(&mut turn, construction_rows.saturating_sub(1), cooperation).await?;
            for row in (1..construction_rows).rev() {
                let draw = draws[construction_rows - 1 - row];
                let selected = ((draw * (row + 1) as f64).floor() as usize).min(row);
                permutation.swap(row, selected);
                cooperation.completed(1).await?;
            }
            drop(draws);
            let jitter = uniforms(&mut turn, construction_rows, cooperation).await?;
            for row in 0..construction_rows {
                remaining[row][column] =
                    (permutation[row] as f64 + jitter[row]) / construction_rows as f64;
                cooperation.completed(1).await?;
            }
        }
        drop(turn);
        Ok(Self::Latin { remaining })
    }

    pub(super) async fn next(
        &mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        match self {
            Self::Latin { remaining } => Ok(remaining.pop_front()),
            Self::Halton { bases, shift, next } => {
                let Some(ordinal) = *next else {
                    return Ok(None);
                };
                let mut point = capacity(bases.len())?;
                for (&base, &rotation) in bases.iter().zip(shift.iter()) {
                    let mut value = ordinal;
                    let mut factor = 1.0 / base as f64;
                    let mut inverse = 0.0;
                    while value != 0 {
                        inverse += (value % base) as f64 * factor;
                        value /= base;
                        factor /= base as f64;
                        cooperation.completed(1).await?;
                    }
                    point.push((inverse + rotation).fract());
                }
                *next = ordinal.checked_add(1);
                Ok(Some(point))
            }
        }
    }
}
