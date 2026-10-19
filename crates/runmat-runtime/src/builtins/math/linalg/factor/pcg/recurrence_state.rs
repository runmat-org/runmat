//! Single PCG recurrence for synchronous and asynchronous operator drivers.
//! All vector storage is borrowed from the caller's admitted workspace.

use super::async_driver::{PcgContinuation, PcgOperation, PcgRecurrence};
use super::kernel::{PcgOutcome, PcgScalar, PcgWork};
use super::scale::ScaledPositive;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

type Checkpoint<'a> = &'a mut dyn FnMut() -> Result<(), RuntimeError>;

#[derive(Clone, Copy)]
struct Seed {
    rhs_squared: ScaledPositive,
    best_squared: ScaledPositive,
}

#[derive(Clone, Copy)]
struct Iteration {
    rhs_squared: ScaledPositive,
    rho: ScaledPositive,
    best_squared: ScaledPositive,
    best_iteration: usize,
    index: usize,
}

#[derive(Clone, Copy)]
enum PreconditionerStage {
    Seed(Seed),
    Following { prior: Iteration },
}

#[derive(Clone, Copy)]
enum SecondInput {
    Residual,
    FirstPreconditioner,
}

#[derive(Clone, Copy)]
enum Phase {
    InitialProduct,
    FirstPreconditioner(PreconditionerStage),
    SecondPreconditioner(PreconditionerStage, SecondInput),
    DirectionProduct(Iteration),
    Complete(PcgOutcome),
}

pub(super) struct RecurrenceState<'a, T> {
    rhs: &'a [T],
    work: PcgWork<'a, T>,
    tolerance: f64,
    maximum_iterations: usize,
    first_preconditioner: bool,
    second_preconditioner: bool,
    phase: Phase,
}

fn invalid(message: &str) -> RuntimeError {
    semantic_error("PcgRecurrence", message)
}

fn copy<T: Copy>(to: &mut [T], from: &[T], checkpoint: Checkpoint<'_>) -> Result<(), RuntimeError> {
    if to.len() != from.len() {
        return Err(invalid("PCG vector extents differ"));
    }
    for (to, from) in to.chunks_mut(1024).zip(from.chunks(1024)) {
        checkpoint()?;
        to.copy_from_slice(from);
    }
    Ok(())
}

struct ScaledDot {
    real: f64,
    imaginary: f64,
    absolute_sum: f64,
    left: f64,
    right: f64,
}

impl ScaledDot {
    fn positive(&self) -> Option<ScaledPositive> {
        ScaledPositive::from_factors(self.real, self.left, self.right)
    }

    fn positive_hermitian<T: PcgScalar>(&self) -> Option<ScaledPositive> {
        let hermitian = self.imaginary.is_finite()
            && self.imaginary.abs() <= 64.0 * T::HERMITIAN_EPSILON * self.absolute_sum;
        self.positive()
            .filter(|positive| hermitian && !positive.is_zero())
    }
}

/// Scale-separated Hermitian product shared by both drivers.
fn dot<T: PcgScalar>(
    left: &[T],
    right: &[T],
    checkpoint: Checkpoint<'_>,
) -> Result<ScaledDot, RuntimeError> {
    if left.len() != right.len() {
        return Err(invalid("PCG dot-product extents differ"));
    }
    let mut left_scale: f64 = 0.0;
    let mut right_scale: f64 = 0.0;
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        if index % 1024 == 0 {
            checkpoint()?;
        }
        left_scale = left_scale.max(a.magnitude());
        right_scale = right_scale.max(b.magnitude());
    }
    if !left_scale.is_finite() || !right_scale.is_finite() {
        return Err(invalid("PCG dot-product magnitude is nonfinite"));
    }
    if left_scale == 0.0 || right_scale == 0.0 {
        return Ok(ScaledDot {
            real: 0.0,
            imaginary: 0.0,
            absolute_sum: 0.0,
            left: left_scale,
            right: right_scale,
        });
    }
    let (mut real, mut imaginary, mut absolute_sum) = (0.0, 0.0, 0.0);
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        if index % 1024 == 0 {
            checkpoint()?;
        }
        let (r, i) = a.scaled_conjugate_product(b, left_scale, right_scale);
        real += r;
        imaginary += i;
        absolute_sum += r.hypot(i);
    }
    Ok(ScaledDot {
        real,
        imaginary,
        absolute_sum,
        left: left_scale,
        right: right_scale,
    })
}

impl<'a, T: PcgScalar> RecurrenceState<'a, T> {
    pub(super) fn new(
        rhs: &'a [T],
        initial: Option<&[T]>,
        tolerance: f64,
        maximum_iterations: usize,
        work: PcgWork<'a, T>,
        preconditioners: (bool, bool),
        checkpoint: Checkpoint<'_>,
    ) -> Result<Self, RuntimeError> {
        let (first_preconditioner, second_preconditioner) = preconditioners;
        let n = rhs.len();
        let residual_capacity = maximum_iterations
            .checked_add(1)
            .ok_or_else(|| invalid("PCG iteration extent overflowed"))?;
        if n == 0
            || maximum_iterations == 0
            || work.x.len() != n
            || work.best.len() != n
            || work.residual.len() != n
            || work.direction.len() != n
            || (first_preconditioner && second_preconditioner)
                && work
                    .intermediate
                    .as_ref()
                    .is_none_or(|values| values.len() != n)
            || work
                .intermediate
                .as_ref()
                .is_some_and(|values| values.len() != n)
            || work.residual_norms.len() < residual_capacity
            || initial.is_some_and(|values| values.len() != n)
        {
            return Err(invalid(
                "PCG work or initial-vector dimensions do not match b",
            ));
        }
        checkpoint()?;
        match initial {
            Some(initial) => copy(&mut *work.x, initial, &mut *checkpoint)?,
            None => {
                for chunk in work.x.chunks_mut(1024) {
                    checkpoint()?;
                    chunk.fill(T::zero());
                }
            }
        }
        Ok(Self {
            rhs,
            work,
            tolerance,
            maximum_iterations,
            first_preconditioner,
            second_preconditioner,
            phase: Phase::InitialProduct,
        })
    }

    fn complete_best(
        &mut self,
        flag: u8,
        iteration: Iteration,
        residual_count: usize,
        checkpoint: Checkpoint<'_>,
    ) -> Result<(), RuntimeError> {
        copy(&mut *self.work.x, &*self.work.best, &mut *checkpoint)?;
        self.phase = Phase::Complete(PcgOutcome {
            flag,
            relres: iteration.best_squared.relative_norm(iteration.rhs_squared),
            iterations: iteration.best_iteration,
            residual_count,
        });
        Ok(())
    }

    fn after_initial_product(
        &mut self,
        output: &[T],
        checkpoint: Checkpoint<'_>,
    ) -> Result<(), RuntimeError> {
        for index in 0..self.rhs.len() {
            if index % 1024 == 0 {
                checkpoint()?;
            }
            self.work.residual[index] = self.rhs[index];
            self.work.residual[index].subtract_scaled(1.0, output[index]);
        }
        let rhs_squared = dot(self.rhs, self.rhs, &mut *checkpoint)?
            .positive()
            .ok_or_else(|| invalid("PCG right-hand-side norm is invalid"))?;
        let best_squared = dot(&*self.work.residual, &*self.work.residual, &mut *checkpoint)?
            .positive()
            .ok_or_else(|| invalid("PCG initial residual norm is invalid"))?;
        self.work.residual_norms[0] = best_squared.norm();
        copy(&mut *self.work.best, &*self.work.x, &mut *checkpoint)?;
        let relres = best_squared.relative_norm(rhs_squared);
        if relres <= self.tolerance {
            self.phase = Phase::Complete(PcgOutcome {
                flag: 0,
                relres,
                iterations: 0,
                residual_count: 1,
            });
            return Ok(());
        }
        let seed = Seed {
            rhs_squared,
            best_squared,
        };
        self.phase = if self.first_preconditioner {
            Phase::FirstPreconditioner(PreconditionerStage::Seed(seed))
        } else if self.second_preconditioner {
            Phase::SecondPreconditioner(PreconditionerStage::Seed(seed), SecondInput::Residual)
        } else {
            copy(
                &mut *self.work.direction,
                &*self.work.residual,
                &mut *checkpoint,
            )?;
            Phase::DirectionProduct(Iteration {
                rhs_squared,
                rho: best_squared,
                best_squared,
                best_iteration: 0,
                index: 0,
            })
        };
        Ok(())
    }

    fn after_preconditioner(
        &mut self,
        stage: PreconditionerStage,
        output: &[T],
        checkpoint: Checkpoint<'_>,
    ) -> Result<(), RuntimeError> {
        let product = dot(&*self.work.residual, output, &mut *checkpoint)?;
        let Some(next_rho) = product.positive_hermitian::<T>() else {
            let iteration = match stage {
                PreconditionerStage::Seed(seed) => Iteration {
                    rhs_squared: seed.rhs_squared,
                    rho: seed.best_squared,
                    best_squared: seed.best_squared,
                    best_iteration: 0,
                    index: 0,
                },
                PreconditionerStage::Following { prior } => prior,
            };
            let residual_count = match stage {
                PreconditionerStage::Seed(_) => 1,
                PreconditionerStage::Following { prior } => prior.index + 2,
            };
            return self.complete_best(4, iteration, residual_count, &mut *checkpoint);
        };
        let next = match stage {
            PreconditionerStage::Seed(seed) => {
                copy(&mut *self.work.direction, output, &mut *checkpoint)?;
                Iteration {
                    rhs_squared: seed.rhs_squared,
                    rho: next_rho,
                    best_squared: seed.best_squared,
                    best_iteration: 0,
                    index: 0,
                }
            }
            PreconditionerStage::Following { prior } => {
                let Some(beta) = next_rho.ratio(prior.rho) else {
                    let residual_count = prior.index + 2;
                    return self.complete_best(4, prior, residual_count, &mut *checkpoint);
                };
                for index in 0..self.rhs.len() {
                    if index % 1024 == 0 {
                        checkpoint()?;
                    }
                    let previous = self.work.direction[index];
                    self.work.direction[index] = output[index];
                    self.work.direction[index].add_scaled(beta, previous);
                }
                Iteration {
                    rhs_squared: prior.rhs_squared,
                    rho: next_rho,
                    best_squared: prior.best_squared,
                    best_iteration: prior.best_iteration,
                    index: prior.index + 1,
                }
            }
        };
        self.phase = Phase::DirectionProduct(next);
        Ok(())
    }

    fn after_direction_product(
        &mut self,
        mut iteration: Iteration,
        output: &[T],
        checkpoint: Checkpoint<'_>,
    ) -> Result<(), RuntimeError> {
        let curvature = dot(&*self.work.direction, output, &mut *checkpoint)?;
        let Some(alpha) = curvature
            .positive_hermitian::<T>()
            .and_then(|curvature| iteration.rho.ratio(curvature))
        else {
            let residual_count = iteration.index + 1;
            return self.complete_best(4, iteration, residual_count, &mut *checkpoint);
        };
        let mut progressed = false;
        for index in 0..self.rhs.len() {
            if index % 1024 == 0 {
                checkpoint()?;
            }
            let previous = self.work.x[index];
            self.work.x[index].add_scaled(alpha, self.work.direction[index]);
            progressed |= self.work.x[index] != previous;
            self.work.residual[index].subtract_scaled(alpha, output[index]);
        }
        let Some(residual_squared) =
            dot(&*self.work.residual, &*self.work.residual, &mut *checkpoint)?.positive()
        else {
            let residual_count = iteration.index + 1;
            return self.complete_best(4, iteration, residual_count, &mut *checkpoint);
        };
        self.work.residual_norms[iteration.index + 1] = residual_squared.norm();
        let relres = residual_squared.relative_norm(iteration.rhs_squared);
        if residual_squared.less_than(iteration.best_squared) {
            iteration.best_squared = residual_squared;
            iteration.best_iteration = iteration.index + 1;
            copy(&mut *self.work.best, &*self.work.x, &mut *checkpoint)?;
        }
        if relres <= self.tolerance {
            self.phase = Phase::Complete(PcgOutcome {
                flag: 0,
                relres,
                iterations: iteration.index + 1,
                residual_count: iteration.index + 2,
            });
            return Ok(());
        }
        if !progressed || iteration.rho.is_zero() {
            return self.complete_best(
                if progressed { 4 } else { 3 },
                iteration,
                iteration.index + 2,
                &mut *checkpoint,
            );
        }
        if self.first_preconditioner || self.second_preconditioner {
            // A final iteration never invokes another user preconditioner.
            if iteration.index + 1 == self.maximum_iterations {
                return self.complete_best(
                    1,
                    iteration,
                    self.maximum_iterations + 1,
                    &mut *checkpoint,
                );
            }
            let stage = PreconditionerStage::Following { prior: iteration };
            self.phase = if self.first_preconditioner {
                Phase::FirstPreconditioner(stage)
            } else {
                Phase::SecondPreconditioner(stage, SecondInput::Residual)
            };
            return Ok(());
        }
        let Some(beta) = residual_squared.ratio(iteration.rho) else {
            let residual_count = iteration.index + 2;
            return self.complete_best(4, iteration, residual_count, &mut *checkpoint);
        };
        for index in 0..self.rhs.len() {
            if index % 1024 == 0 {
                checkpoint()?;
            }
            let previous = self.work.direction[index];
            self.work.direction[index] = self.work.residual[index];
            self.work.direction[index].add_scaled(beta, previous);
        }
        // The synchronous kernel computes beta and updates the direction even
        // at the limit, so its breakdown disposition remains unchanged.
        if iteration.index + 1 == self.maximum_iterations {
            return self.complete_best(1, iteration, self.maximum_iterations + 1, &mut *checkpoint);
        }
        self.phase = Phase::DirectionProduct(Iteration {
            rhs_squared: iteration.rhs_squared,
            rho: residual_squared,
            best_squared: iteration.best_squared,
            best_iteration: iteration.best_iteration,
            index: iteration.index + 1,
        });
        Ok(())
    }
}

impl<T: PcgScalar> PcgRecurrence<T> for RecurrenceState<'_, T> {
    fn continuation(&self) -> PcgContinuation<'_, T> {
        match self.phase {
            Phase::InitialProduct => PcgContinuation::Apply {
                operation: PcgOperation::Coefficient,
                input: &*self.work.x,
            },
            Phase::FirstPreconditioner(_) => PcgContinuation::Apply {
                operation: PcgOperation::FirstPreconditioner,
                input: &*self.work.residual,
            },
            Phase::SecondPreconditioner(_, SecondInput::Residual) => PcgContinuation::Apply {
                operation: PcgOperation::SecondPreconditioner,
                input: &*self.work.residual,
            },
            Phase::SecondPreconditioner(_, SecondInput::FirstPreconditioner) => {
                PcgContinuation::Apply {
                    operation: PcgOperation::SecondPreconditioner,
                    input: self
                        .work
                        .intermediate
                        .as_deref()
                        .expect("two-stage preconditioner owns its intermediate"),
                }
            }
            Phase::DirectionProduct(_) => PcgContinuation::Apply {
                operation: PcgOperation::Coefficient,
                input: &*self.work.direction,
            },
            Phase::Complete(outcome) => PcgContinuation::Complete(outcome),
        }
    }

    fn accept(&mut self, output: &[T], checkpoint: Checkpoint<'_>) -> Result<(), RuntimeError> {
        if output.len() != self.rhs.len() {
            return Err(invalid("PCG operation returned the wrong vector extent"));
        }
        match self.phase {
            Phase::InitialProduct => self.after_initial_product(output, checkpoint),
            Phase::FirstPreconditioner(stage) => {
                if self.second_preconditioner {
                    copy(
                        self.work
                            .intermediate
                            .as_deref_mut()
                            .expect("two-stage preconditioner owns its intermediate"),
                        output,
                        &mut *checkpoint,
                    )?;
                    self.phase =
                        Phase::SecondPreconditioner(stage, SecondInput::FirstPreconditioner);
                    Ok(())
                } else {
                    self.after_preconditioner(stage, output, checkpoint)
                }
            }
            Phase::SecondPreconditioner(stage, _) => {
                self.after_preconditioner(stage, output, checkpoint)
            }
            Phase::DirectionProduct(iteration) => {
                self.after_direction_product(iteration, output, checkpoint)
            }
            Phase::Complete(_) => Err(invalid("PCG recurrence accepted a result after completion")),
        }
    }
}

/// Synchronous driver for the existing dense/CSC operator route. The supplied
/// operation buffer is separately admitted by the caller; no callback result
/// may borrow it after the next transition.
pub(super) fn drive_sync<T: PcgScalar>(
    mut recurrence: RecurrenceState<'_, T>,
    operation_scratch: &mut [T],
    mut apply: impl FnMut(PcgOperation, &[T], &mut [T]) -> Result<(), RuntimeError>,
    mut checkpoint: impl FnMut() -> Result<(), RuntimeError>,
) -> Result<PcgOutcome, RuntimeError> {
    loop {
        checkpoint()?;
        let (operation, input) = match recurrence.continuation() {
            PcgContinuation::Apply { operation, input } => (operation, input),
            PcgContinuation::Complete(outcome) => return Ok(outcome),
        };
        if input.len() != operation_scratch.len() {
            return Err(invalid(
                "PCG operation scratch extent does not match its input",
            ));
        }
        apply(operation, input, operation_scratch)?;
        checkpoint()?;
        recurrence.accept(operation_scratch, &mut checkpoint)?;
    }
}
