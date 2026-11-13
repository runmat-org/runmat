//! Restarted GMRES: the one Arnoldi recurrence of the crate. All arrays are
//! supplied by the caller; this module never allocates or owns an operator.
//!
//! [`GmresMode`] selects where the preconditioner acts and when the true
//! residual is formed. The default (left preconditioning, a true residual
//! after every Arnoldi step) is the MATLAB-compatible behavior of the
//! runtime's `gmres`. Right and flexible preconditioning minimize the true
//! residual `‖b − A x‖`; [`ResidualCheck::OnEstimate`] forms it only when the
//! Arnoldi estimate passes or a cycle ends, and still certifies convergence on
//! that true residual before returning flag 0.

use super::usage::{BatchCompletion, GivensBranch, ScalarWork, VectorWork};
use num_complex::Complex64;

use super::{ApplyStatus as PreconditionerStatus, KrylovFailure};
mod scalar;
use scalar::{axpy, copy, divide, dot, fill, norm, residual, Norm};
mod geometry;
pub use geometry::GmresGeometry;

pub trait GmresOperations<T> {
    type Error;

    /// Admit a typed arithmetic batch before its synchronous numerical work.
    /// `completed` is false when an optional native kernel declined the work.
    fn measured<R>(
        &mut self,
        work: super::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, super::usage::BatchCompletion),
    ) -> impl std::future::Future<Output = Result<R, Self::Error>>;

    fn coefficient(
        &mut self,
        input: &[T],
        output: &mut [T],
    ) -> impl std::future::Future<Output = Result<(), Self::Error>>;
    fn precondition(
        &mut self,
        input: &[T],
        output: &mut [T],
        intermediate: &mut [T],
    ) -> impl std::future::Future<Output = Result<PreconditionerStatus, Self::Error>>;
    fn checkpoint(&mut self) -> Result<(), Self::Error>;
    fn cooperate(
        &mut self,
        completed: u64,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>>;

    // Optional native vector kernels. A caller whose storage has native, deterministic (and,
    // for a partitioned backend, globally reduced) kernels supplies them; returning
    // `None`/`false` (the default) selects the recurrence's portable scale-separated kernels
    // for that call. A native result that is not finite also returns `None`, so extreme-range
    // vectors take the portable path. Every native answer must be identical on all ranks of a
    // partitioned solve.

    /// Euclidean norm, finite.
    fn native_norm(&mut self, _values: &[T]) -> Option<f64> {
        None
    }
    /// Hermitian product `Σ conj(left_i) right_i`, finite.
    fn native_dot(&mut self, _left: &[T], _right: &[T]) -> Option<Complex64> {
        None
    }
    /// `target += scale · source`; `false` leaves `target` untouched.
    fn native_axpy(&mut self, _target: &mut [T], _scale: Complex64, _source: &[T]) -> bool {
        false
    }
    /// `target = source / divisor` for a finite positive divisor; `false` leaves `target`
    /// untouched.
    fn native_divide(&mut self, _target: &mut [T], _source: &[T], _divisor: f64) -> bool {
        false
    }
    /// One classical Gram–Schmidt pass against the `rows` contiguous basis vectors:
    /// `coefficients[k] = ⟨v_k, w⟩` for all `k`, then `w −= Σ coefficients[k] v_k`. Returning
    /// `false` must leave `w` untouched and selects the portable sequential projection.
    fn native_project(
        &mut self,
        _basis: &[T],
        _rows: usize,
        _w: &mut [T],
        _coefficients: &mut [Complex64],
    ) -> bool {
        false
    }
    /// `residual = rhs − product`: `Some(true)` when every entry is representable,
    /// `Some(false)` when one is not (breakdown), `None` for the portable loop.
    fn native_residual(&mut self, _rhs: &[T], _product: &[T], _residual: &mut [T]) -> Option<bool> {
        None
    }
    /// Whether two vectors are equal entry by entry.
    fn native_equal(&mut self, _left: &[T], _right: &[T]) -> Option<bool> {
        None
    }
}

pub struct GmresWork<'a, T> {
    pub x: &'a mut [T],
    pub best: &'a mut [T],
    pub candidate: &'a mut [T],
    pub previous: &'a mut [T],
    pub residual: &'a mut [T],
    pub product: &'a mut [T],
    pub preconditioned: &'a mut [T],
    pub intermediate: &'a mut [T],
    pub basis: &'a mut [T],
    pub hessenberg: &'a mut [Complex64],
    pub cosines: &'a mut [f64],
    pub sines: &'a mut [Complex64],
    pub rotated_rhs: &'a mut [Complex64],
    pub coefficients: &'a mut [Complex64],
    pub history: &'a mut [f64],
}

#[derive(Clone, Copy, Debug)]
pub struct GmresOutcome {
    pub flag: u8,
    pub relres: f64,
    pub outer: usize,
    pub inner: usize,
    pub residual_count: usize,
    /// Arnoldi steps performed (each one coefficient application).
    pub steps: usize,
}

/// Where the preconditioner `M` acts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PreconditionerSide {
    /// Minimizes `‖M⁻¹(b − A x)‖` (MATLAB `gmres`).
    #[default]
    Left,
    /// Minimizes the true residual `‖b − A M⁻¹ u‖`; the correction is `M⁻¹ V y`.
    Right,
    /// Right preconditioning that keeps `Z = M⁻¹ V` in caller workspace, so `M`
    /// may change between applications (inner iterative solves, variable cycles).
    Flexible,
}

/// When the residual of the candidate iterate is formed explicitly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResidualCheck {
    /// After every Arnoldi step (one extra coefficient application, and for
    /// left/right preconditioning one extra preconditioner application, per step).
    #[default]
    EveryStep,
    /// Only when the Arnoldi residual estimate meets the tolerance, the Krylov
    /// space becomes invariant, or the cycle ends. The candidate then becomes the
    /// iterate and the restart forms its residual; flag 0 is returned only when
    /// that formed residual meets the tolerance, otherwise iteration continues.
    OnEstimate,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GmresMode {
    pub side: PreconditionerSide,
    pub residual_check: ResidualCheck,
    /// [`ResidualCheck::OnEstimate`] only: an exact bound on Arnoldi steps. The cycle that
    /// reaches it ends there and its iterate is verified like any other.
    pub step_limit: Option<usize>,
}

#[derive(Clone, Copy)]
pub struct GmresParameters {
    pub restart: usize,
    pub max_outer: usize,
    pub tolerance: f64,
}

/// A zero-sized native arithmetic policy avoids imposing a foreign scalar
/// trait on a caller's canonical storage type. No view cast or copy is needed.
///
/// The optional vector kernels let a policy whose storage has native, deterministic
/// reductions supply them. Returning `None`/`false` (the default) selects the
/// recurrence's portable scale-separated kernels for that call; a policy also
/// returns `None` whenever its native result is not finite, so extreme-range
/// vectors still take the portable path.
pub trait Arithmetic<T> {
    const DOMAIN: super::usage::ArithmeticDomain;
    fn zero() -> T;
    fn to_complex(value: T) -> Complex64;
    fn from_complex(value: Complex64) -> Option<T>;
    fn divide_complex(numerator: (f64, f64), denominator: (f64, f64)) -> (f64, f64);
}

/// Left-preconditioned GMRES with a true residual after every step.
/// Check every workspace region before the first operation/control effect.
pub async fn solve<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    rhs: &[T],
    initial: Option<&[T]>,
    parameters: GmresParameters,
    work: &mut GmresWork<'_, T>,
    operations: &mut O,
) -> Result<GmresOutcome, KrylovFailure<O::Error>> {
    solve_with::<T, A, O>(
        rhs,
        initial,
        parameters,
        GmresMode::default(),
        work,
        &mut [],
        operations,
    )
    .await
}

/// GMRES in the selected [`GmresMode`]. `flexible_basis` holds `Z = M⁻¹ V`
/// ([`GmresGeometry::flexible_elements`] entries) and may be empty otherwise.
pub async fn solve_with<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    rhs: &[T],
    initial: Option<&[T]>,
    parameters: GmresParameters,
    mode: GmresMode,
    work: &mut GmresWork<'_, T>,
    flexible_basis: &mut [T],
    operations: &mut O,
) -> Result<GmresOutcome, KrylovFailure<O::Error>> {
    let geometry = GmresGeometry::checked(rhs.len(), parameters.restart, parameters.max_outer)
        .ok_or(KrylovFailure::Invalid(
            "GMRES workspace geometry overflows or is empty",
        ))?;
    if !parameters.tolerance.is_finite()
        || parameters.tolerance <= 0.0
        || initial.is_some_and(|x| x.len() != rhs.len())
        || !geometry.validates(work)
        || (mode.side == PreconditionerSide::Flexible
            && flexible_basis.len() < geometry.flexible_elements())
        || mode
            .step_limit
            .is_some_and(|limit| limit == 0 || mode.residual_check != ResidualCheck::OnEstimate)
    {
        return Err(KrylovFailure::Invalid(
            "GMRES parameters or workspace extents are invalid",
        ));
    }
    solve_validated::<T, A, O>(
        rhs,
        initial,
        parameters,
        mode,
        work,
        flexible_basis,
        operations,
    )
    .await
    .map_err(KrylovFailure::Operation)
}

fn h_index(row: usize, column: usize, restart: usize) -> usize {
    column * (restart + 1) + row
}

mod algebra;
pub use algebra::rotation;
use algebra::{apply_previous_rotations, candidate_from_basis, BasisProjection, Candidate};

/// The caller has checked every length and owns the exact physical workspaces.
/// A callback error is terminal; only numerical breakdown is a solver flag.
async fn solve_validated<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    rhs: &[T],
    initial: Option<&[T]>,
    parameters: GmresParameters,
    mode: GmresMode,
    work: &mut GmresWork<'_, T>,
    flexible_basis: &mut [T],
    operations: &mut O,
) -> Result<GmresOutcome, O::Error> {
    let GmresParameters {
        restart,
        max_outer,
        tolerance,
    } = parameters;
    let n = rhs.len();
    let side = mode.side;
    let on_estimate = mode.residual_check == ResidualCheck::OnEstimate;
    let step_limit = mode.step_limit.unwrap_or(usize::MAX);
    let mut steps = 0usize;
    // Restart position of the current iterate x.
    let mut x_position = (0, 0);
    operations.checkpoint()?;
    if let Some(initial) = initial {
        copy(initial, work.x, operations).await?;
    } else {
        fill(work.x, A::zero(), operations).await?;
    }
    copy(work.x, work.best, operations).await?;
    let reference_source: &[T] = if side == PreconditionerSide::Left {
        if operations
            .precondition(rhs, work.preconditioned, work.intermediate)
            .await?
            == PreconditionerStatus::NumericalFailure
        {
            let (outer, inner) = (0, 0);
            return Ok(GmresOutcome {
                flag: 2,
                relres: f64::INFINITY,
                outer,
                inner,
                residual_count: 0,
                steps,
            });
        }
        &*work.preconditioned
    } else {
        rhs
    };
    let Some(reference) = norm::<T, A, O>(reference_source, operations).await? else {
        return Ok(GmresOutcome {
            flag: 4,
            relres: f64::INFINITY,
            outer: 0,
            inner: 0,
            residual_count: 0,
            steps,
        });
    };
    let denominator = if reference.is_zero() {
        Norm::one()
    } else {
        reference
    };
    let mut best_residual = Norm::infinity();
    let mut best_iteration = (0, 0);
    let mut residual_count = 0;
    let mut previous_candidate = false;
    // OnEstimate verifies the last cycle's iterate in one further restart pass.
    for outer in 0..max_outer + usize::from(on_estimate) {
        operations.checkpoint()?;
        for chunk in work.hessenberg[..(restart + 1) * restart].chunks_mut(1024) {
            operations.cooperate(chunk.len() as u64).await?;
            chunk.fill(Complex64::new(0.0, 0.0));
        }
        operations.coefficient(work.x, work.product).await?;
        if !residual::<T, A, O>(rhs, work.product, work.residual, operations).await? {
            return Ok(GmresOutcome {
                flag: 4,
                relres: best_residual
                    .measured_relative(denominator, operations)
                    .await?,
                outer: best_iteration.0,
                inner: best_iteration.1,
                residual_count,
                steps,
            });
        }
        if side == PreconditionerSide::Left {
            if operations
                .precondition(work.residual, work.preconditioned, work.intermediate)
                .await?
                == PreconditionerStatus::NumericalFailure
            {
                let (outer, inner) = best_iteration;
                return Ok(GmresOutcome {
                    flag: 2,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer,
                    inner,
                    residual_count,
                    steps,
                });
            }
        } else {
            copy(work.residual, work.preconditioned, operations).await?;
        }
        let Some(beta) = norm::<T, A, O>(work.preconditioned, operations).await? else {
            return Ok(GmresOutcome {
                flag: 4,
                relres: best_residual
                    .measured_relative(denominator, operations)
                    .await?,
                outer: best_iteration.0,
                inner: best_iteration.1,
                residual_count,
                steps,
            });
        };
        if beta.measured_less_than(best_residual, operations).await? {
            best_residual = beta;
            if outer > 0 {
                best_iteration = x_position;
            }
            copy(work.x, work.best, operations).await?;
        }
        if residual_count == 0 {
            work.history[0] = beta.measured_magnitude(operations).await?;
            residual_count = 1;
        }
        if beta.measured_relative(denominator, operations).await? <= tolerance {
            let (outer, inner) = if on_estimate { x_position } else { (outer, 0) };
            return Ok(GmresOutcome {
                flag: 0,
                relres: beta.measured_relative(denominator, operations).await?,
                outer,
                inner,
                residual_count,
                steps,
            });
        }
        if outer == max_outer || steps >= step_limit {
            // OnEstimate verification pass after the last permitted cycle or step.
            break;
        }
        if !divide::<T, A, O>(work.preconditioned, &mut work.basis[..n], beta, operations).await? {
            return Ok(GmresOutcome {
                flag: 4,
                relres: best_residual
                    .measured_relative(denominator, operations)
                    .await?,
                outer: best_iteration.0,
                inner: best_iteration.1,
                residual_count,
                steps,
            });
        }
        for chunk in work.rotated_rhs[..restart + 1].chunks_mut(1024) {
            operations.cooperate(chunk.len() as u64).await?;
            chunk.fill(Complex64::new(0.0, 0.0));
        }
        // Preserve the ordinary absolute-RHS solve when beta fits: a tiny
        // H can have representable beta/H even though 1/H would overflow.
        // Only an unrepresentable absolute beta uses unit g with the residual
        // scale carried through the subsequent native candidate correction.
        let absolute_beta = beta.measured_magnitude(operations).await?;
        let (projected_rhs, correction_scale) = if absolute_beta.is_finite() {
            (absolute_beta, Norm::one())
        } else {
            (1.0, beta)
        };
        work.rotated_rhs[0] = Complex64::new(projected_rhs, 0.0);
        for inner in 0..restart {
            operations.checkpoint()?;
            let basis = &work.basis[inner * n..(inner + 1) * n];
            let status = match side {
                PreconditionerSide::Left => {
                    operations.coefficient(basis, work.product).await?;
                    operations
                        .precondition(work.product, work.preconditioned, work.intermediate)
                        .await?
                }
                PreconditionerSide::Right => {
                    let status = operations
                        .precondition(basis, work.product, work.intermediate)
                        .await?;
                    if status == PreconditionerStatus::Applied {
                        operations
                            .coefficient(work.product, work.preconditioned)
                            .await?;
                    }
                    status
                }
                PreconditionerSide::Flexible => {
                    let direction = &mut flexible_basis[inner * n..(inner + 1) * n];
                    let status = operations
                        .precondition(basis, direction, work.intermediate)
                        .await?;
                    if status == PreconditionerStatus::Applied {
                        operations
                            .coefficient(direction, work.preconditioned)
                            .await?;
                    }
                    status
                }
            };
            steps += 1;
            if status == PreconditionerStatus::NumericalFailure {
                return Ok(GmresOutcome {
                    flag: 2,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            }
            let Some(before) = norm::<T, A, O>(work.preconditioned, operations).await? else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            };
            for pass in 0..2 {
                let rows = inner + 1;
                let native_projected = operations
                    .measured(
                        VectorWork::Projection {
                            rows,
                            elements: n,
                            domain: A::DOMAIN,
                        },
                        |operations| {
                            let applied = operations.native_project(
                                &work.basis[..rows * n],
                                rows,
                                work.preconditioned,
                                &mut work.coefficients[..rows],
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
                    .await?;
                if native_projected {
                    operations.cooperate((2 * rows * n) as u64).await?;
                    for row in 0..rows {
                        operations
                            .measured(VectorWork::HessenbergAdd { domain: A::DOMAIN }, |_| {
                                work.hessenberg[h_index(row, inner, restart)] +=
                                    work.coefficients[row];
                                ((), BatchCompletion::Complete)
                            })
                            .await?;
                    }
                } else {
                    for row in 0..=inner {
                        let basis = &work.basis[row * n..(row + 1) * n];
                        let Some(coefficient) =
                            dot::<T, A, O>(basis, work.preconditioned, operations).await?
                        else {
                            return Ok(GmresOutcome {
                                flag: 4,
                                relres: best_residual
                                    .measured_relative(denominator, operations)
                                    .await?,
                                outer: best_iteration.0,
                                inner: best_iteration.1,
                                residual_count,
                                steps,
                            });
                        };
                        operations
                            .measured(VectorWork::HessenbergAdd { domain: A::DOMAIN }, |_| {
                                work.hessenberg[h_index(row, inner, restart)] += coefficient;
                                ((), BatchCompletion::Complete)
                            })
                            .await?;
                        if !axpy::<T, A, O>(work.preconditioned, -coefficient, basis, operations)
                            .await?
                        {
                            return Ok(GmresOutcome {
                                flag: 4,
                                relres: best_residual
                                    .measured_relative(denominator, operations)
                                    .await?,
                                outer: best_iteration.0,
                                inner: best_iteration.1,
                                residual_count,
                                steps,
                            });
                        }
                    }
                }
                let reorthogonalized = if pass == 0 {
                    match norm::<T, A, O>(work.preconditioned, operations).await? {
                        Some(after) => after.measured_relative(before, operations).await? >= 0.717,
                        None => false,
                    }
                } else {
                    false
                };
                if reorthogonalized {
                    break;
                }
            }
            let Some(next_norm) = norm::<T, A, O>(work.preconditioned, operations).await? else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            };
            work.hessenberg[h_index(inner + 1, inner, restart)] =
                Complex64::new(next_norm.measured_magnitude(operations).await?, 0.0);
            if !next_norm.is_zero()
                && !divide::<T, A, O>(
                    work.preconditioned,
                    &mut work.basis[(inner + 1) * n..(inner + 2) * n],
                    next_norm,
                    operations,
                )
                .await?
            {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            }
            apply_previous_rotations(work, inner, restart, operations).await?;
            let first = h_index(inner, inner, restart);
            let second = first + 1;
            let a = work.hessenberg[first];
            let b = work.hessenberg[second];
            let a_zero = a.re == 0.0 && a.im == 0.0;
            let b_zero = b.re == 0.0 && b.im == 0.0;
            let givens = VectorWork::Givens {
                magnitude_evaluations: u8::from(a.im != 0.0) + u8::from(b.im != 0.0),
                branch: if a_zero && b_zero {
                    GivensBranch::Zero
                } else if a_zero {
                    GivensBranch::ZeroFirst
                } else {
                    GivensBranch::General
                },
            };
            let rotated = operations
                .measured(givens, |_| {
                    let result = rotation(a, b);
                    (
                        result,
                        if result.is_some() {
                            BatchCompletion::Complete
                        } else {
                            BatchCompletion::Skipped
                        },
                    )
                })
                .await?;
            let Some((c, s)) = rotated else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            };
            work.cosines[inner] = c;
            work.sines[inner] = s;
            operations
                .measured(
                    VectorWork::Scalar(ScalarWork::ComplexRotationFinalize),
                    |_| {
                        work.hessenberg[first] = c * a + s * b;
                        work.hessenberg[second] = Complex64::new(0.0, 0.0);
                        let g = work.rotated_rhs[inner];
                        work.rotated_rhs[inner] = c * g;
                        work.rotated_rhs[inner + 1] = -s.conj() * g;
                        ((), BatchCompletion::Complete)
                    },
                )
                .await?;
            if on_estimate {
                let rotated = work.rotated_rhs[inner + 1];
                let rotated_magnitude = operations
                    .measured(
                        VectorWork::Scalar(if rotated.im == 0.0 {
                            ScalarWork::Zero
                        } else {
                            ScalarWork::Hypot
                        }),
                        |_| {
                            (
                                if rotated.im == 0.0 {
                                    rotated.re.abs()
                                } else {
                                    rotated.norm()
                                },
                                BatchCompletion::Complete,
                            )
                        },
                    )
                    .await?;
                let estimate = operations
                    .measured(
                        VectorWork::Scalar(
                            if rotated_magnitude == 0.0 || correction_scale.is_zero() {
                                ScalarWork::Zero
                            } else {
                                ScalarWork::NormScale
                            },
                        ),
                        |_| {
                            (
                                correction_scale.times(rotated_magnitude),
                                BatchCompletion::Complete,
                            )
                        },
                    )
                    .await?;
                work.history[residual_count] = operations
                    .measured(VectorWork::Scalar(ScalarWork::NormScale), |_| {
                        (estimate.magnitude(), BatchCompletion::Complete)
                    })
                    .await?;
                residual_count += 1;
                let relative_estimate = operations
                    .measured(VectorWork::Scalar(estimate.relative_work()), |_| {
                        (estimate.relative(denominator), BatchCompletion::Complete)
                    })
                    .await?;
                if relative_estimate > tolerance
                    && !next_norm.is_zero()
                    && inner + 1 < restart
                    && steps < step_limit
                {
                    continue;
                }
            }
            match candidate_from_basis::<T, A, O>(
                work,
                BasisProjection {
                    completed: inner + 1,
                    restart,
                    dimension: n,
                },
                correction_scale,
                side,
                flexible_basis,
                operations,
            )
            .await?
            {
                Candidate::Ready => {}
                Candidate::Breakdown => {
                    return Ok(GmresOutcome {
                        flag: 4,
                        relres: best_residual
                            .measured_relative(denominator, operations)
                            .await?,
                        outer: best_iteration.0,
                        inner: best_iteration.1,
                        residual_count,
                        steps,
                    });
                }
                Candidate::PreconditionerFailure => {
                    return Ok(GmresOutcome {
                        flag: 2,
                        relres: best_residual
                            .measured_relative(denominator, operations)
                            .await?,
                        outer: best_iteration.0,
                        inner: best_iteration.1,
                        residual_count,
                        steps,
                    });
                }
            }
            if on_estimate {
                // The candidate becomes the iterate; the restart forms and
                // certifies its residual. An unchanged iterate is stagnation.
                let same = match operations.native_equal(work.candidate, work.x) {
                    Some(same) => same,
                    None => {
                        let mut same = true;
                        for (candidate, current) in
                            work.candidate.chunks(1024).zip(work.x.chunks(1024))
                        {
                            operations.cooperate(candidate.len() as u64).await?;
                            same &= candidate == current;
                        }
                        same
                    }
                };
                if same {
                    copy(work.best, work.x, operations).await?;
                    return Ok(GmresOutcome {
                        flag: 3,
                        relres: best_residual
                            .measured_relative(denominator, operations)
                            .await?,
                        outer: best_iteration.0,
                        inner: best_iteration.1,
                        residual_count,
                        steps,
                    });
                }
                copy(work.candidate, work.x, operations).await?;
                x_position = (outer + 1, inner + 1);
                break;
            }
            operations.coefficient(work.candidate, work.product).await?;
            if !residual::<T, A, O>(rhs, work.product, work.residual, operations).await? {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            }
            let checked: &[T] = if side == PreconditionerSide::Left {
                if operations
                    .precondition(work.residual, work.preconditioned, work.intermediate)
                    .await?
                    == PreconditionerStatus::NumericalFailure
                {
                    return Ok(GmresOutcome {
                        flag: 2,
                        relres: best_residual
                            .measured_relative(denominator, operations)
                            .await?,
                        outer: best_iteration.0,
                        inner: best_iteration.1,
                        residual_count,
                        steps,
                    });
                }
                &*work.preconditioned
            } else {
                &*work.residual
            };
            let Some(actual) = norm::<T, A, O>(checked, operations).await? else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            };
            work.history[residual_count] = actual.measured_magnitude(operations).await?;
            residual_count += 1;
            let last = (outer + 1, inner + 1);
            if actual.measured_less_than(best_residual, operations).await? {
                best_residual = actual;
                best_iteration = last;
                copy(work.candidate, work.best, operations).await?;
            }
            if actual.measured_relative(denominator, operations).await? <= tolerance {
                copy(work.candidate, work.x, operations).await?;
                return Ok(GmresOutcome {
                    flag: 0,
                    relres: actual.measured_relative(denominator, operations).await?,
                    outer: outer + 1,
                    inner: inner + 1,
                    residual_count,
                    steps,
                });
            }
            let same = match operations.native_equal(work.candidate, work.previous) {
                Some(equal) => previous_candidate && equal,
                None => {
                    let mut same = previous_candidate;
                    for (candidate, previous) in
                        work.candidate.chunks(1024).zip(work.previous.chunks(1024))
                    {
                        operations.cooperate(candidate.len() as u64).await?;
                        same &= candidate == previous;
                    }
                    same
                }
            };
            if same {
                return Ok(GmresOutcome {
                    flag: 3,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            }
            copy(work.candidate, work.previous, operations).await?;
            previous_candidate = true;
            if next_norm.is_zero() {
                // The Krylov space is invariant but its best candidate did
                // not satisfy the true residual test. Restarting the same
                // space cannot create a new direction; report breakdown,
                // retaining the best independently checked iterate.
                copy(work.best, work.x, operations).await?;
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual
                        .measured_relative(denominator, operations)
                        .await?,
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                    steps,
                });
            }
            if inner + 1 == restart {
                copy(work.candidate, work.x, operations).await?;
                x_position = (outer + 1, restart);
            }
        }
    }
    copy(work.best, work.x, operations).await?;
    Ok(GmresOutcome {
        flag: 1,
        relres: best_residual
            .measured_relative(denominator, operations)
            .await?,
        outer: best_iteration.0,
        inner: best_iteration.1,
        residual_count,
        steps,
    })
}
