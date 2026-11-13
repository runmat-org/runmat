//! Accepted-iterate inverse updates. Limited-memory storage owns only the
//! retained secant pairs, never a dense inverse Hessian.
use super::algorithm::vectors::{copied as copy, dot as inner};
use super::algorithm::{identity, mat_vec};
use super::*;
use crate::builtins::math::optim::common::ScalarCooperation;
use runmat_builtins::OptimQuasiNewtonUpdate as Update;
mod dense;
#[cfg(test)]
mod tests;

fn allocation(source: std::collections::TryReserveError) -> RuntimeError {
    build_runtime_error("fminunc: L-BFGS storage admission failed")
        .with_builtin(NAME)
        .with_identifier("RunMat:fminunc:AllocationFailed")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}

async fn axpy(
    output: &mut [f64],
    input: &[f64],
    scale: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    for (output, input) in output.chunks_mut(1024).zip(input.chunks(1024)) {
        for (out, value) in output.iter_mut().zip(input) {
            *out += scale * value;
        }
        cooperation.completed(output.len() as u64).await?;
    }
    Ok(())
}

pub(super) fn controls(
    fields: &crate::builtins::math::optim::common::ScalarOptionFields<'_>,
) -> BuiltinResult<Update> {
    use runmat_builtins::OptimOptionField as Field;
    let parse = |field| {
        fields
            .exact(field)
            .map(|value| {
                crate::builtins::math::optim::common::quasi_newton_controls::update(
                    NAME, field, value,
                )
            })
            .transpose()
    };
    let modern = parse(Field::HessianApproximation)?;
    let legacy = parse(Field::HessUpdate)?;
    if let (Some(a), Some(b)) = (modern, legacy) {
        if a != b {
            return Err(fminunc_error_with_detail(
                &FMINUNC_ERROR_INVALID_ARGUMENT,
                "HessianApproximation conflicts with legacy HessUpdate",
            ));
        }
    }
    Ok(modern.or(legacy).unwrap_or(Update::Bfgs))
}

pub(super) fn hessian_unavailable(detail: &str) -> RuntimeError {
    build_runtime_error(format!(
        "fminunc: requested final Hessian unavailable: {detail}"
    ))
    .with_builtin(NAME)
    .with_identifier("RunMat:fminunc:HessianUnavailable")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

pub(super) enum Model {
    Dense {
        values: Vec<f64>,
        update: Update,
    },
    Limited {
        pairs: std::collections::VecDeque<Pair>,
        limit: usize,
    },
    Steepest,
}
pub(super) struct Pair {
    s: Vec<f64>,
    y: Vec<f64>,
    rho: f64,
}
impl Model {
    pub(super) fn new(update: Update, n: usize, max_iter: usize) -> BuiltinResult<Self> {
        Ok(match update {
            Update::Bfgs | Update::Dfp => {
                validate_problem_size(n)?;
                Self::Dense {
                    values: identity(n)?,
                    update,
                }
            }
            Update::LimitedMemory { history } => Self::Limited {
                pairs: std::collections::VecDeque::new(),
                // There cannot be more stored updates than accepted iterations.
                // This does not change the requested history's mathematics.
                limit: history.get().min(max_iter),
            },
            Update::Steepest => Self::Steepest,
        })
    }
    pub(super) async fn reset(
        &mut self,
        n: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        match self {
            Self::Dense { values, .. } => {
                for chunk in values.chunks_mut(1024) {
                    chunk.fill(0.0);
                    cooperation.completed(chunk.len() as u64).await?;
                }
                for index in 0..n {
                    values[index * n + index] = 1.0;
                }
            }
            Self::Limited { pairs, .. } => {
                while let Some(pair) = pairs.pop_front() {
                    drop(pair);
                    cooperation.completed(n as u64).await?;
                }
            }
            Self::Steepest => {}
        }
        Ok(())
    }
    pub(super) async fn direction(
        &self,
        gradient: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<f64>> {
        let mut result = match self {
            Self::Dense { values, .. } => {
                mat_vec(values, gradient, gradient.len(), cooperation).await?
            }
            Self::Steepest => copy(gradient, cooperation).await?,
            Self::Limited { pairs, .. } => {
                let mut q = copy(gradient, cooperation).await?;
                let mut alpha = Vec::new();
                alpha.try_reserve_exact(pairs.len()).map_err(allocation)?;
                for pair in pairs.iter().rev() {
                    let a = pair.rho * inner(&pair.s, &q, cooperation).await?;
                    alpha.push(a);
                    axpy(&mut q, &pair.y, -a, cooperation).await?;
                }
                if let Some(last) = pairs.back() {
                    let scale = inner(&last.s, &last.y, cooperation).await?
                        / inner(&last.y, &last.y, cooperation).await?;
                    for chunk in q.chunks_mut(1024) {
                        for value in &mut *chunk {
                            *value *= scale;
                        }
                        cooperation.completed(chunk.len() as u64).await?;
                    }
                }
                for (pair, a) in pairs.iter().zip(alpha.into_iter().rev()) {
                    let beta = pair.rho * inner(&pair.y, &q, cooperation).await?;
                    axpy(&mut q, &pair.s, a - beta, cooperation).await?;
                }
                q
            }
        };
        for chunk in result.chunks_mut(1024) {
            for value in &mut *chunk {
                *value = -*value;
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
        Ok(result)
    }
    pub(super) async fn update(
        &mut self,
        s: Vec<f64>,
        y: Vec<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        match self {
            Self::Dense { values, update } => {
                dense::update(values, &s, &y, *update, cooperation).await?
            }
            Self::Limited { pairs, limit } if *limit != 0 => {
                let sy = inner(&s, &y, cooperation).await?;
                let yy = inner(&y, &y, cooperation).await?;
                if sy > 0.0
                    && sy.is_finite()
                    && yy > 0.0
                    && yy.is_finite()
                    && (1.0 / sy).is_finite()
                {
                    // Admit metadata before relinquishing an old pair. Payloads
                    // are the actual accepted differences, moved exactly once.
                    if pairs.len() < *limit {
                        pairs.try_reserve(1).map_err(allocation)?;
                    }
                    if pairs.len() == *limit {
                        pairs.pop_front();
                    }
                    pairs.push_back(Pair {
                        s,
                        y,
                        rho: 1.0 / sy,
                    });
                }
            }
            Self::Limited { .. } | Self::Steepest => {}
        }
        Ok(())
    }
}
