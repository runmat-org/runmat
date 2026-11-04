//! Reliable-residual PCG state over borrowed admitted slices.
use super::scale::ScaledPositive;
use super::{Arithmetic, PcgContinuation, PcgOperation, PcgOperations, PcgOutcome, PcgWork};
use crate::krylov::usage::{BatchCompletion, ScalarWork, VectorWork};
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
    /// Flexible mode: the residual before the latest update (Polak–Ribière β).
    previous: Option<&'a mut [T]>,
    arithmetic: std::marker::PhantomData<A>,
}

impl<T, A> RecurrenceState<'_, T, A> {
    fn record_beta(&mut self, beta: f64) {
        if let Some(coefficients) = self.work.coefficients.as_deref_mut() {
            coefficients.beta.push(beta);
        }
    }

    /// A reliable residual restart starts a new Lanczos segment.
    fn restart_coefficients(&mut self) {
        if let Some(coefficients) = self.work.coefficients.as_deref_mut() {
            coefficients.restart();
        }
    }
}

fn invalid<E>(message: &'static str) -> KrylovFailure<E> {
    KrylovFailure::Invalid(message)
}

async fn measured_norm<T, C: PcgOperations<T>>(
    value: ScaledPositive,
    checkpoint: &mut C,
) -> Result<f64, KrylovFailure<C::Error>> {
    checkpoint
        .measured(
            VectorWork::Scalar(if value.is_zero() {
                ScalarWork::Zero
            } else {
                ScalarWork::Norm
            }),
            |_| (value.norm(), BatchCompletion::Complete),
        )
        .await
        .map_err(KrylovFailure::Operation)
}

async fn measured_relative_norm<T, C: PcgOperations<T>>(
    numerator: ScaledPositive,
    denominator: ScaledPositive,
    checkpoint: &mut C,
) -> Result<f64, KrylovFailure<C::Error>> {
    let work = if numerator.is_zero() {
        ScalarWork::Zero
    } else if denominator.is_zero() {
        ScalarWork::Norm
    } else {
        ScalarWork::RelativeNorm
    };
    checkpoint
        .measured(VectorWork::Scalar(work), |_| {
            (
                numerator.relative_norm(denominator),
                BatchCompletion::Complete,
            )
        })
        .await
        .map_err(KrylovFailure::Operation)
}

async fn measured_ratio<T, C: PcgOperations<T>>(
    numerator: ScaledPositive,
    denominator: ScaledPositive,
    checkpoint: &mut C,
) -> Result<Option<f64>, KrylovFailure<C::Error>> {
    let work = if denominator.is_zero() {
        ScalarWork::Zero
    } else {
        ScalarWork::Ratio
    };
    checkpoint
        .measured(VectorWork::Scalar(work), |_| {
            (numerator.ratio(denominator), BatchCompletion::Complete)
        })
        .await
        .map_err(KrylovFailure::Operation)
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
    /// The real part as a plain number (flexible β; ranges where it overflows stop the
    /// recurrence as a breakdown).
    fn value(&self) -> f64 {
        self.real * self.left * self.right
    }
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
    let native = checkpoint
        .measured(
            VectorWork::Dot {
                elements: left.len(),
                domain: A::DOMAIN,
            },
            |checkpoint| {
                let result = checkpoint.native_dot(left, right);
                (
                    result,
                    if result.is_some() {
                        BatchCompletion::Complete
                    } else {
                        BatchCompletion::Skipped
                    },
                )
            },
        )
        .await
        .map_err(KrylovFailure::Operation)?;
    if let Some((real, imaginary, absolute_sum)) = native {
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
    for start in (0..left.len()).step_by(1024) {
        let end = (start + 1024).min(left.len());
        let mut terms = Vec::with_capacity(end - start);
        checkpoint
            .cooperate((end - start) as u64)
            .await
            .map_err(KrylovFailure::Operation)?;
        checkpoint
            .measured(
                if start == 0 {
                    VectorWork::Dot {
                        elements: end - start,
                        domain: A::DOMAIN,
                    }
                } else {
                    VectorWork::DotContinue {
                        elements: end - start,
                        domain: A::DOMAIN,
                    }
                },
                |_| {
                    for index in start..end {
                        let (r, i) = A::scaled_conjugate_product(
                            left[index],
                            right[index],
                            left_scale,
                            right_scale,
                        );
                        real += r;
                        imaginary += i;
                        terms.push((r, i));
                    }
                    ((), BatchCompletion::Complete)
                },
            )
            .await
            .map_err(KrylovFailure::Operation)?;
        let complex_entries = terms
            .iter()
            .filter(|(_, imaginary)| *imaginary != 0.0)
            .count();
        checkpoint
            .measured(
                VectorWork::MagnitudeAccumulate {
                    elements: terms.len(),
                    complex_entries,
                },
                |_| {
                    for (real, imaginary) in &terms {
                        absolute_sum += if *imaginary == 0.0 {
                            real.abs()
                        } else {
                            real.hypot(*imaginary)
                        };
                    }
                    ((), BatchCompletion::Complete)
                },
            )
            .await
            .map_err(KrylovFailure::Operation)?;
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
        parameters: super::PcgParameters,
        work: PcgWork<'a, T>,
        previous: Option<&'a mut [T]>,
        checkpoint: &mut C,
    ) -> Result<Self, KrylovFailure<C::Error>> {
        let super::PcgParameters {
            tolerance,
            maximum_iterations,
            preconditioners,
        } = parameters;
        let (first_preconditioner, second_preconditioner) = preconditioners;
        let mut work = work;
        if let Some(coefficients) = work.coefficients.as_deref_mut() {
            coefficients.restart();
            coefficients.flexible = previous.is_some();
        }
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
            previous,
            arithmetic: std::marker::PhantomData,
        })
    }

    async fn after_initial_product<C: PcgOperations<T>>(
        &mut self,
        output: &[T],
        checkpoint: &mut C,
    ) -> Result<(), KrylovFailure<C::Error>> {
        let native = checkpoint
            .measured(
                VectorWork::Residual {
                    elements: self.rhs.len(),
                    domain: A::DOMAIN,
                },
                |checkpoint| {
                    let applied = checkpoint.native_residual(
                        self.rhs,
                        &output[..self.rhs.len()],
                        &mut *self.work.residual,
                    );
                    (
                        applied,
                        if applied {
                            BatchCompletion::Complete
                        } else {
                            BatchCompletion::Skipped
                        },
                    )
                },
            )
            .await
            .map_err(KrylovFailure::Operation)?;
        if native {
            checkpoint
                .cooperate(self.rhs.len() as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        } else {
            for start in (0..self.rhs.len()).step_by(1024) {
                let end = (start + 1024).min(self.rhs.len());
                checkpoint
                    .cooperate((end - start) as u64)
                    .await
                    .map_err(KrylovFailure::Operation)?;
                checkpoint
                    .measured(
                        VectorWork::Residual {
                            elements: end - start,
                            domain: A::DOMAIN,
                        },
                        |_| {
                            for (offset, product) in output[start..end].iter().enumerate() {
                                let index = start + offset;
                                self.work.residual[index] = self.rhs[index];
                                A::subtract_scaled(&mut self.work.residual[index], 1.0, *product);
                            }
                            ((), BatchCompletion::Complete)
                        },
                    )
                    .await
                    .map_err(KrylovFailure::Operation)?;
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
        self.work.residual_norms[0] = measured_norm::<T, C>(best_squared, checkpoint).await?;
        copy(&mut *self.work.best, &*self.work.x, &mut *checkpoint).await?;
        self.best_certificate = Some(best_squared);
        let relres = measured_relative_norm::<T, C>(best_squared, rhs_squared, checkpoint).await?;
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
        let next_rho = checkpoint
            .measured(VectorWork::Scalar(ScalarWork::HermitianCheck), |_| {
                (
                    product.positive_hermitian::<T, A>(),
                    BatchCompletion::Complete,
                )
            })
            .await
            .map_err(KrylovFailure::Operation)?;
        let Some(next_rho) = next_rho else {
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
                let beta = match self.previous.as_deref() {
                    // Flexible: β = (⟨r, z⟩ − ⟨r_old, z⟩) / ⟨r_old, z_old⟩.
                    Some(previous) => {
                        let cross = dot::<T, A, C>(previous, output, &mut *checkpoint).await?;
                        let value = checkpoint
                            .measured(VectorWork::Scalar(ScalarWork::FlexibleRatio), |_| {
                                (
                                    (next_rho.value() - cross.value()) / prior.rho.value(),
                                    BatchCompletion::Complete,
                                )
                            })
                            .await
                            .map_err(KrylovFailure::Operation)?;
                        value.is_finite().then_some(value)
                    }
                    None => measured_ratio::<T, C>(next_rho, prior.rho, checkpoint).await?,
                };
                let Some(beta) = beta else {
                    let residual_count = prior.index + 2;
                    return self
                        .complete_best(4, prior, residual_count, &mut *checkpoint)
                        .await;
                };
                self.record_beta(beta);
                let native = checkpoint
                    .measured(
                        VectorWork::Direction {
                            elements: self.rhs.len(),
                            domain: A::DOMAIN,
                        },
                        |checkpoint| {
                            let applied = checkpoint.native_direction(
                                &mut *self.work.direction,
                                &output[..self.rhs.len()],
                                beta,
                            );
                            (
                                applied,
                                if applied {
                                    BatchCompletion::Complete
                                } else {
                                    BatchCompletion::Skipped
                                },
                            )
                        },
                    )
                    .await
                    .map_err(KrylovFailure::Operation)?;
                if native {
                    checkpoint
                        .cooperate(self.rhs.len() as u64)
                        .await
                        .map_err(KrylovFailure::Operation)?;
                } else {
                    for start in (0..self.rhs.len()).step_by(1024) {
                        let end = (start + 1024).min(self.rhs.len());
                        checkpoint
                            .cooperate((end - start) as u64)
                            .await
                            .map_err(KrylovFailure::Operation)?;
                        checkpoint
                            .measured(
                                VectorWork::Direction {
                                    elements: end - start,
                                    domain: A::DOMAIN,
                                },
                                |_| {
                                    for (offset, product) in output[start..end].iter().enumerate() {
                                        let index = start + offset;
                                        let previous = self.work.direction[index];
                                        self.work.direction[index] = *product;
                                        A::add_scaled(
                                            &mut self.work.direction[index],
                                            beta,
                                            previous,
                                        );
                                    }
                                    ((), BatchCompletion::Complete)
                                },
                            )
                            .await
                            .map_err(KrylovFailure::Operation)?;
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
                self.restart_coefficients();
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
        let positive_curvature = checkpoint
            .measured(VectorWork::Scalar(ScalarWork::HermitianCheck), |_| {
                (
                    curvature.positive_hermitian::<T, A>(),
                    BatchCompletion::Complete,
                )
            })
            .await
            .map_err(KrylovFailure::Operation)?;
        let alpha = match positive_curvature {
            Some(curvature) => measured_ratio::<T, C>(iteration.rho, curvature, checkpoint).await?,
            None => None,
        };
        let Some(alpha) = alpha else {
            let residual_count = iteration.index + 1;
            return self
                .complete_best(4, iteration, residual_count, &mut *checkpoint)
                .await;
        };
        if let Some(coefficients) = self.work.coefficients.as_deref_mut() {
            coefficients.alpha.push(alpha);
        }
        if let Some(previous) = self.previous.as_deref_mut() {
            copy(previous, &*self.work.residual, &mut *checkpoint).await?;
        }
        let mut progressed = false;
        let native = checkpoint
            .measured(
                VectorWork::Step {
                    elements: self.rhs.len(),
                    domain: A::DOMAIN,
                },
                |checkpoint| {
                    let result = checkpoint.native_step(
                        &mut *self.work.x,
                        &mut *self.work.residual,
                        &*self.work.direction,
                        &output[..self.rhs.len()],
                        alpha,
                    );
                    (
                        result,
                        if result.is_some() {
                            BatchCompletion::Complete
                        } else {
                            BatchCompletion::Skipped
                        },
                    )
                },
            )
            .await
            .map_err(KrylovFailure::Operation)?;
        if let Some(native) = native {
            progressed = native;
            checkpoint
                .cooperate(self.rhs.len() as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        } else {
            for start in (0..self.rhs.len()).step_by(1024) {
                let end = (start + 1024).min(self.rhs.len());
                checkpoint
                    .cooperate((end - start) as u64)
                    .await
                    .map_err(KrylovFailure::Operation)?;
                let changed = checkpoint
                    .measured(
                        VectorWork::Step {
                            elements: end - start,
                            domain: A::DOMAIN,
                        },
                        |_| {
                            let mut changed = false;
                            for (offset, product) in output[start..end].iter().enumerate() {
                                let index = start + offset;
                                let previous = self.work.x[index];
                                A::add_scaled(
                                    &mut self.work.x[index],
                                    alpha,
                                    self.work.direction[index],
                                );
                                changed |= self.work.x[index] != previous;
                                A::subtract_scaled(&mut self.work.residual[index], alpha, *product);
                            }
                            (changed, BatchCompletion::Complete)
                        },
                    )
                    .await
                    .map_err(KrylovFailure::Operation)?;
                progressed |= changed;
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
        self.work.residual_norms[iteration.index + 1] =
            measured_norm::<T, C>(residual_squared, checkpoint).await?;
        let relres =
            measured_relative_norm::<T, C>(residual_squared, iteration.rhs_squared, checkpoint)
                .await?;
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
        let Some(beta) =
            measured_ratio::<T, C>(residual_squared, iteration.rho, checkpoint).await?
        else {
            let residual_count = iteration.index + 2;
            return self
                .complete_best(4, iteration, residual_count, &mut *checkpoint)
                .await;
        };
        self.record_beta(beta);
        let native = checkpoint
            .measured(
                VectorWork::Direction {
                    elements: self.rhs.len(),
                    domain: A::DOMAIN,
                },
                |checkpoint| {
                    let applied = checkpoint.native_direction(
                        &mut *self.work.direction,
                        &*self.work.residual,
                        beta,
                    );
                    (
                        applied,
                        if applied {
                            BatchCompletion::Complete
                        } else {
                            BatchCompletion::Skipped
                        },
                    )
                },
            )
            .await
            .map_err(KrylovFailure::Operation)?;
        if native {
            checkpoint
                .cooperate(self.rhs.len() as u64)
                .await
                .map_err(KrylovFailure::Operation)?;
        } else {
            for start in (0..self.rhs.len()).step_by(1024) {
                let end = (start + 1024).min(self.rhs.len());
                checkpoint
                    .cooperate((end - start) as u64)
                    .await
                    .map_err(KrylovFailure::Operation)?;
                checkpoint
                    .measured(
                        VectorWork::Direction {
                            elements: end - start,
                            domain: A::DOMAIN,
                        },
                        |_| {
                            for index in start..end {
                                let previous = self.work.direction[index];
                                self.work.direction[index] = self.work.residual[index];
                                A::add_scaled(&mut self.work.direction[index], beta, previous);
                            }
                            ((), BatchCompletion::Complete)
                        },
                    )
                    .await
                    .map_err(KrylovFailure::Operation)?;
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
