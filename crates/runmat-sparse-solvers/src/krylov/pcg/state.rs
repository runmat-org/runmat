//! Reliable-residual PCG state over borrowed admitted slices.
use super::scale::ScaledPositive;
use super::{Arithmetic, PcgContinuation, PcgOperation, PcgOperations, PcgOutcome, PcgWork};
use crate::krylov::KrylovFailure;
mod completion;

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
    Restart { prior: Iteration },
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
    CandidateProduct(Iteration, bool),
    BestProduct(u8, Iteration, usize),
    Complete(PcgOutcome),
}

pub(super) struct RecurrenceState<'a, T, A> {
    rhs: &'a [T],
    work: PcgWork<'a, T>,
    tolerance: f64,
    maximum_iterations: usize,
    first_preconditioner: bool,
    second_preconditioner: bool,
    phase: Phase,
    best_certificate: Option<ScaledPositive>,
    arithmetic: std::marker::PhantomData<A>,
}

fn invalid<E>(message: &'static str) -> KrylovFailure<E> {
    KrylovFailure::Invalid(message)
}

async fn copy<T: Copy, C: PcgOperations<T>>(
    to: &mut [T],
    from: &[T],
    checkpoint: &mut C,
) -> Result<(), KrylovFailure<C::Error>> {
    if to.len() != from.len() {
        return Err(invalid("PCG vector extents differ"));
    }
    for (to, from) in to.chunks_mut(1024).zip(from.chunks(1024)) {
        checkpoint
            .cooperate(to.len() as u64)
            .await
            .map_err(KrylovFailure::Operation)?;
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

    fn positive_hermitian<T, A: Arithmetic<T>>(&self) -> Option<ScaledPositive> {
        let hermitian = self.imaginary.is_finite()
            && self.imaginary.abs() <= 64.0 * A::HERMITIAN_EPSILON * self.absolute_sum;
        self.positive()
            .filter(|positive| hermitian && !positive.is_zero())
    }
}

/// Scale-separated Hermitian product shared by both drivers.
async fn dot<T: Copy + PartialEq, A: Arithmetic<T>, C: PcgOperations<T>>(
    left: &[T],
    right: &[T],
    checkpoint: &mut C,
) -> Result<ScaledDot, KrylovFailure<C::Error>> {
    if left.len() != right.len() {
        return Err(invalid("PCG dot-product extents differ"));
    }
    if let Some((real, imaginary, absolute_sum)) = checkpoint.native_dot(left, right) {
        checkpoint
            .cooperate(left.len() as u64)
            .await
            .map_err(KrylovFailure::Operation)?;
        return Ok(ScaledDot {
            real,
            imaginary,
            absolute_sum,
            left: 1.0,
            right: 1.0,
        });
    }
    let mut left_scale: f64 = 0.0;
    let mut right_scale: f64 = 0.0;
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        if index % 1024 == 0 {
            checkpoint
                .cooperate((left.len() - index).min(1024) as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        }
        let (ar, ai) = A::components(a);
        let (br, bi) = A::components(b);
        if !ar.is_finite() || !ai.is_finite() || !br.is_finite() || !bi.is_finite() {
            return Err(invalid("PCG dot-product operand is nonfinite"));
        }
        left_scale = left_scale.max(ar.abs()).max(ai.abs());
        right_scale = right_scale.max(br.abs()).max(bi.abs());
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
            checkpoint
                .cooperate((left.len() - index).min(1024) as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        }
        let (r, i) = A::scaled_conjugate_product(a, b, left_scale, right_scale);
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

impl<'a, T: Copy + PartialEq, A: Arithmetic<T>> RecurrenceState<'a, T, A> {
    pub(super) async fn new<C: PcgOperations<T>>(
        rhs: &'a [T],
        initial: Option<&[T]>,
        tolerance: f64,
        maximum_iterations: usize,
        work: PcgWork<'a, T>,
        preconditioners: (bool, bool),
        checkpoint: &mut C,
    ) -> Result<Self, KrylovFailure<C::Error>> {
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
        checkpoint.checkpoint().map_err(KrylovFailure::Operation)?;
        match initial {
            Some(initial) => copy(&mut *work.x, initial, &mut *checkpoint).await?,
            None => {
                for chunk in work.x.chunks_mut(1024) {
                    checkpoint
                        .cooperate(chunk.len() as u64)
                        .await
                        .map_err(KrylovFailure::Operation)?;
                    chunk.fill(A::zero());
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
            best_certificate: None,
            arithmetic: std::marker::PhantomData,
        })
    }

    async fn after_initial_product<C: PcgOperations<T>>(
        &mut self,
        output: &[T],
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        if checkpoint.native_residual(self.rhs, &output[..self.rhs.len()], &mut *self.work.residual) {
            checkpoint
                .cooperate(self.rhs.len() as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        } else {
            for (index, &value) in output[..self.rhs.len()].iter().enumerate() {
                if index % 1024 == 0 {
                    checkpoint
                        .cooperate((self.rhs.len() - index).min(1024) as u64)
                        .await
                        .map_err(KrylovFailure::Operation)?;
                }
                self.work.residual[index] = self.rhs[index];
                A::subtract_scaled(&mut self.work.residual[index], 1.0, value);
            }
        }
        let rhs_squared = dot::<T, A, C>(self.rhs, self.rhs, &mut *checkpoint)
            .await?
            .positive()
            .ok_or_else(|| invalid("PCG right-hand-side norm is invalid"))?;
        let best_squared =
            dot::<T, A, C>(&*self.work.residual, &*self.work.residual, &mut *checkpoint)
                .await?
                .positive()
                .ok_or_else(|| invalid("PCG initial residual norm is invalid"))?;
        self.work.residual_norms[0] = best_squared.norm();
        copy(&mut *self.work.best, &*self.work.x, &mut *checkpoint).await?;
        self.best_certificate = Some(best_squared);
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
            )
            .await?;
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

    async fn after_preconditioner<C: PcgOperations<T>>(
        &mut self,
        stage: PreconditionerStage,
        output: &[T],
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        let product = dot::<T, A, C>(&*self.work.residual, output, &mut *checkpoint).await?;
        let Some(next_rho) = product.positive_hermitian::<T, A>() else {
            let iteration = match stage {
                PreconditionerStage::Seed(seed) => Iteration {
                    rhs_squared: seed.rhs_squared,
                    rho: seed.best_squared,
                    best_squared: seed.best_squared,
                    best_iteration: 0,
                    index: 0,
                },
                PreconditionerStage::Following { prior }
                | PreconditionerStage::Restart { prior } => prior,
            };
            let residual_count = match stage {
                PreconditionerStage::Seed(_) => 1,
                PreconditionerStage::Following { prior }
                | PreconditionerStage::Restart { prior } => prior.index + 2,
            };
            return self
                .complete_best(4, iteration, residual_count, &mut *checkpoint)
                .await;
        };
        let next = match stage {
            PreconditionerStage::Seed(seed) => {
                copy(&mut *self.work.direction, output, &mut *checkpoint).await?;
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
                    return self
                        .complete_best(4, prior, residual_count, &mut *checkpoint)
                        .await;
                };
                if checkpoint.native_direction(&mut *self.work.direction, &output[..self.rhs.len()], beta) {
                    checkpoint
                        .cooperate(self.rhs.len() as u64)
                        .await
                        .map_err(KrylovFailure::Operation)?;
                } else {
                    for (index, &value) in output[..self.rhs.len()].iter().enumerate() {
                        if index % 1024 == 0 {
                            checkpoint
                                .cooperate((self.rhs.len() - index).min(1024) as u64)
                                .await
                                .map_err(KrylovFailure::Operation)?;
                        }
                        let previous = self.work.direction[index];
                        self.work.direction[index] = value;
                        A::add_scaled(&mut self.work.direction[index], beta, previous);
                    }
                }
                Iteration {
                    rhs_squared: prior.rhs_squared,
                    rho: next_rho,
                    best_squared: prior.best_squared,
                    best_iteration: prior.best_iteration,
                    index: prior.index + 1,
                }
            }
            PreconditionerStage::Restart { prior } => {
                copy(&mut *self.work.direction, output, &mut *checkpoint).await?;
                Iteration {
                    rho: next_rho,
                    index: prior.index + 1,
                    ..prior
                }
            }
        };
        self.phase = Phase::DirectionProduct(next);
        Ok(())
    }

    async fn after_direction_product<C: PcgOperations<T>>(
        &mut self,
        mut iteration: Iteration,
        output: &[T],
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        let curvature = dot::<T, A, C>(&*self.work.direction, output, &mut *checkpoint).await?;
        let Some(alpha) = curvature
            .positive_hermitian::<T, A>()
            .and_then(|curvature| iteration.rho.ratio(curvature))
        else {
            let residual_count = iteration.index + 1;
            return self
                .complete_best(4, iteration, residual_count, &mut *checkpoint)
                .await;
        };
        let mut progressed = false;
        if let Some(native) = checkpoint.native_step(
            &mut *self.work.x,
            &mut *self.work.residual,
            &*self.work.direction,
            &output[..self.rhs.len()],
            alpha,
        ) {
            progressed = native;
            checkpoint
                .cooperate(self.rhs.len() as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        } else {
            for (index, &value) in output[..self.rhs.len()].iter().enumerate() {
                if index % 1024 == 0 {
                    checkpoint
                        .cooperate((self.rhs.len() - index).min(1024) as u64)
                        .await
                        .map_err(KrylovFailure::Operation)?;
                }
                let previous = self.work.x[index];
                A::add_scaled(&mut self.work.x[index], alpha, self.work.direction[index]);
                progressed |= self.work.x[index] != previous;
                A::subtract_scaled(&mut self.work.residual[index], alpha, value);
            }
        }
        let Some(residual_squared) =
            dot::<T, A, C>(&*self.work.residual, &*self.work.residual, &mut *checkpoint)
                .await?
                .positive()
        else {
            let residual_count = iteration.index + 1;
            return self
                .complete_best(4, iteration, residual_count, &mut *checkpoint)
                .await;
        };
        self.work.residual_norms[iteration.index + 1] = residual_squared.norm();
        let relres = residual_squared.relative_norm(iteration.rhs_squared);
        if !progressed {
            // x did not change. In particular, do not replay a successful
            // certificate of this already retained best iterate.
            return self
                .complete_best(3, iteration, iteration.index + 2, &mut *checkpoint)
                .await;
        }
        if relres <= self.tolerance {
            // A new request for A*x certifies the rounded candidate. This is
            // not a replay of A*p and does not double ordinary iteration work.
            self.phase = Phase::CandidateProduct(iteration, progressed);
            return Ok(());
        }
        if residual_squared.less_than(iteration.best_squared) {
            iteration.best_squared = residual_squared;
            iteration.best_iteration = iteration.index + 1;
            copy(&mut *self.work.best, &*self.work.x, &mut *checkpoint).await?;
            self.best_certificate = None;
        }
        if iteration.rho.is_zero() {
            return self
                .complete_best(4, iteration, iteration.index + 2, &mut *checkpoint)
                .await;
        }
        if self.first_preconditioner || self.second_preconditioner {
            // A final iteration never invokes another user preconditioner.
            if iteration.index + 1 == self.maximum_iterations {
                return self
                    .complete_best(1, iteration, self.maximum_iterations + 1, &mut *checkpoint)
                    .await;
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
            return self
                .complete_best(4, iteration, residual_count, &mut *checkpoint)
                .await;
        };
        if checkpoint.native_direction(&mut *self.work.direction, &*self.work.residual, beta) {
            checkpoint
                .cooperate(self.rhs.len() as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        } else {
            for index in 0..self.rhs.len() {
                if index % 1024 == 0 {
                    checkpoint
                        .cooperate((self.rhs.len() - index).min(1024) as u64)
                        .await
                        .map_err(KrylovFailure::Operation)?;
                }
                let previous = self.work.direction[index];
                self.work.direction[index] = self.work.residual[index];
                A::add_scaled(&mut self.work.direction[index], beta, previous);
            }
        }
        // The synchronous kernel computes beta and updates the direction even
        // at the limit, so its breakdown disposition remains unchanged.
        if iteration.index + 1 == self.maximum_iterations {
            return self
                .complete_best(1, iteration, self.maximum_iterations + 1, &mut *checkpoint)
                .await;
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

impl<T: Copy + PartialEq, A: Arithmetic<T>> RecurrenceState<'_, T, A> {
    pub(super) async fn preconditioner_failed<C: PcgOperations<T>>(
        &mut self,
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        self.complete_preconditioner_failure(checkpoint).await
    }

    pub(super) fn continuation(&self) -> PcgContinuation<'_, T> {
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
            Phase::CandidateProduct(_, _) => PcgContinuation::Apply {
                operation: PcgOperation::Coefficient,
                input: &*self.work.x,
            },
            Phase::BestProduct(_, _, _) => PcgContinuation::Apply {
                operation: PcgOperation::Coefficient,
                input: &*self.work.best,
            },
            Phase::Complete(outcome) => PcgContinuation::Complete(outcome),
        }
    }

    pub(super) async fn accept<C: PcgOperations<T>>(
        &mut self,
        output: &[T],
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        if output.len() != self.rhs.len() {
            return Err(invalid("PCG operation returned the wrong vector extent"));
        }
        match self.phase {
            Phase::InitialProduct => self.after_initial_product(output, checkpoint).await,
            Phase::FirstPreconditioner(stage) => {
                if self.second_preconditioner {
                    copy(
                        self.work
                            .intermediate
                            .as_deref_mut()
                            .expect("two-stage preconditioner owns its intermediate"),
                        output,
                        &mut *checkpoint,
                    )
                    .await?;
                    self.phase =
                        Phase::SecondPreconditioner(stage, SecondInput::FirstPreconditioner);
                    Ok(())
                } else {
                    self.after_preconditioner(stage, output, checkpoint).await
                }
            }
            Phase::SecondPreconditioner(stage, _) => {
                self.after_preconditioner(stage, output, checkpoint).await
            }
            Phase::DirectionProduct(iteration) => {
                self.after_direction_product(iteration, output, checkpoint)
                    .await
            }
            Phase::CandidateProduct(iteration, progressed) => {
                self.after_candidate_product(iteration, progressed, output, checkpoint)
                    .await
            }
            Phase::BestProduct(flag, iteration, residual_count) => {
                self.best_certificate = Some(self.true_residual(output, &mut *checkpoint).await?);
                self.complete_best(flag, iteration, residual_count, checkpoint)
                    .await
            }
            Phase::Complete(_) => Err(invalid("PCG recurrence accepted a result after completion")),
        }
    }
}
