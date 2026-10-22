//! Restarted, left-preconditioned Arnoldi recurrence. All arrays are supplied
//! by the caller; this module never allocates or owns an operator.

use num_complex::Complex64;

use super::{ApplyStatus as PreconditionerStatus, KrylovFailure};
mod scalar;
use scalar::{axpy, copy, dot, fill, norm, Norm};
mod geometry;
pub use geometry::GmresGeometry;

pub trait GmresOperations<T> {
    type Error;

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
}

#[derive(Clone, Copy)]
pub struct GmresParameters {
    pub restart: usize,
    pub max_outer: usize,
    pub tolerance: f64,
}

/// A zero-sized native arithmetic policy avoids imposing a foreign scalar
/// trait on a caller's canonical storage type. No view cast or copy is needed.
pub trait Arithmetic<T> {
    fn zero() -> T;
    fn to_complex(value: T) -> Complex64;
    fn from_complex(value: Complex64) -> Option<T>;
    fn divide_complex(numerator: (f64, f64), denominator: (f64, f64)) -> (f64, f64);
}

/// Check every workspace region before the first operation/control effect.
pub async fn solve<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    rhs: &[T],
    initial: Option<&[T]>,
    parameters: GmresParameters,
    work: &mut GmresWork<'_, T>,
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
    {
        return Err(KrylovFailure::Invalid(
            "GMRES parameters or workspace extents are invalid",
        ));
    }
    solve_validated::<T, A, O>(rhs, initial, parameters, work, operations)
        .await
        .map_err(KrylovFailure::Operation)
}

fn h_index(row: usize, column: usize, restart: usize) -> usize {
    column * (restart + 1) + row
}

mod algebra;
pub use algebra::rotation;
use algebra::{apply_previous_rotations, candidate_from_basis};

/// The caller has checked every length and owns the exact physical workspaces.
/// A callback error is terminal; only numerical breakdown is a solver flag.
async fn solve_validated<T: Copy + PartialEq, A: Arithmetic<T>, O: GmresOperations<T>>(
    rhs: &[T],
    initial: Option<&[T]>,
    parameters: GmresParameters,
    work: &mut GmresWork<'_, T>,
    operations: &mut O,
) -> Result<GmresOutcome, O::Error> {
    let GmresParameters {
        restart,
        max_outer,
        tolerance,
    } = parameters;
    let n = rhs.len();
    operations.checkpoint()?;
    if let Some(initial) = initial {
        copy(initial, work.x, operations).await?;
    } else {
        fill(work.x, A::zero(), operations).await?;
    }
    copy(work.x, work.best, operations).await?;
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
        });
    }
    let Some(reference) = norm::<T, A, O>(work.preconditioned, operations).await? else {
        return Ok(GmresOutcome {
            flag: 4,
            relres: f64::INFINITY,
            outer: 0,
            inner: 0,
            residual_count: 0,
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
    for outer in 0..max_outer {
        operations.checkpoint()?;
        for chunk in work.hessenberg[..(restart + 1) * restart].chunks_mut(1024) {
            operations.cooperate(chunk.len() as u64).await?;
            chunk.fill(Complex64::new(0.0, 0.0));
        }
        operations.coefficient(work.x, work.product).await?;
        for (index, (destination, (&b, &ax))) in work
            .residual
            .iter_mut()
            .zip(rhs.iter().zip(&*work.product))
            .enumerate()
        {
            if index % 1024 == 0 {
                operations.cooperate((n - index).min(1024) as u64).await?;
            }
            let Some(value) = A::from_complex(A::to_complex(b) - A::to_complex(ax)) else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            };
            *destination = value;
        }
        if operations
            .precondition(work.residual, work.preconditioned, work.intermediate)
            .await?
            == PreconditionerStatus::NumericalFailure
        {
            let (outer, inner) = best_iteration;
            return Ok(GmresOutcome {
                flag: 2,
                relres: best_residual.relative(denominator),
                outer,
                inner,
                residual_count,
            });
        }
        let Some(beta) = norm::<T, A, O>(work.preconditioned, operations).await? else {
            return Ok(GmresOutcome {
                flag: 4,
                relres: best_residual.relative(denominator),
                outer: best_iteration.0,
                inner: best_iteration.1,
                residual_count,
            });
        };
        if beta.less_than(best_residual) {
            best_residual = beta;
            if outer > 0 {
                best_iteration = (outer, restart);
            }
            copy(work.x, work.best, operations).await?;
        }
        if residual_count == 0 {
            work.history[0] = beta.magnitude();
            residual_count = 1;
        }
        if beta.relative(denominator) <= tolerance {
            return Ok(GmresOutcome {
                flag: 0,
                relres: beta.relative(denominator),
                outer,
                inner: 0,
                residual_count,
            });
        }
        for (index, (target, &value)) in work.basis[..n]
            .iter_mut()
            .zip(&*work.preconditioned)
            .enumerate()
        {
            if index % 1024 == 0 {
                operations.cooperate((n - index).min(1024) as u64).await?;
            }
            let Some(value) = A::from_complex(beta.normalize(A::to_complex(value))) else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            };
            *target = value;
        }
        for chunk in work.rotated_rhs[..restart + 1].chunks_mut(1024) {
            operations.cooperate(chunk.len() as u64).await?;
            chunk.fill(Complex64::new(0.0, 0.0));
        }
        // Preserve the ordinary absolute-RHS solve when beta fits: a tiny
        // H can have representable beta/H even though 1/H would overflow.
        // Only an unrepresentable absolute beta uses unit g with the residual
        // scale carried through the subsequent native candidate correction.
        let absolute_beta = beta.magnitude();
        let (projected_rhs, correction_scale) = if absolute_beta.is_finite() {
            (absolute_beta, Norm::one())
        } else {
            (1.0, beta)
        };
        work.rotated_rhs[0] = Complex64::new(projected_rhs, 0.0);
        for inner in 0..restart {
            operations.checkpoint()?;
            let basis = &work.basis[inner * n..(inner + 1) * n];
            operations.coefficient(basis, work.product).await?;
            if operations
                .precondition(work.product, work.preconditioned, work.intermediate)
                .await?
                == PreconditionerStatus::NumericalFailure
            {
                return Ok(GmresOutcome {
                    flag: 2,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            }
            let Some(before) = norm::<T, A, O>(work.preconditioned, operations).await? else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            };
            for pass in 0..2 {
                for row in 0..=inner {
                    let basis = &work.basis[row * n..(row + 1) * n];
                    let Some(coefficient) =
                        dot::<T, A, O>(basis, work.preconditioned, operations).await?
                    else {
                        return Ok(GmresOutcome {
                            flag: 4,
                            relres: best_residual.relative(denominator),
                            outer: best_iteration.0,
                            inner: best_iteration.1,
                            residual_count,
                        });
                    };
                    work.hessenberg[h_index(row, inner, restart)] += coefficient;
                    if !axpy::<T, A, O>(work.preconditioned, -coefficient, basis, operations)
                        .await?
                    {
                        return Ok(GmresOutcome {
                            flag: 4,
                            relres: best_residual.relative(denominator),
                            outer: best_iteration.0,
                            inner: best_iteration.1,
                            residual_count,
                        });
                    }
                }
                if pass == 0
                    && norm::<T, A, O>(work.preconditioned, operations)
                        .await?
                        .is_some_and(|after| after.relative(before) >= 0.717)
                {
                    break;
                }
            }
            let Some(next_norm) = norm::<T, A, O>(work.preconditioned, operations).await? else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            };
            work.hessenberg[h_index(inner + 1, inner, restart)] =
                Complex64::new(next_norm.magnitude(), 0.0);
            if !next_norm.is_zero() {
                for (index, (target, &value)) in work.basis[(inner + 1) * n..(inner + 2) * n]
                    .iter_mut()
                    .zip(&*work.preconditioned)
                    .enumerate()
                {
                    if index % 1024 == 0 {
                        operations.cooperate((n - index).min(1024) as u64).await?;
                    }
                    let Some(value) = A::from_complex(next_norm.normalize(A::to_complex(value)))
                    else {
                        return Ok(GmresOutcome {
                            flag: 4,
                            relres: best_residual.relative(denominator),
                            outer: best_iteration.0,
                            inner: best_iteration.1,
                            residual_count,
                        });
                    };
                    *target = value;
                }
            }
            apply_previous_rotations(work, inner, restart, operations).await?;
            let first = h_index(inner, inner, restart);
            let second = first + 1;
            let Some((c, s)) = rotation(work.hessenberg[first], work.hessenberg[second]) else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            };
            work.cosines[inner] = c;
            work.sines[inner] = s;
            let a = work.hessenberg[first];
            let b = work.hessenberg[second];
            work.hessenberg[first] = c * a + s * b;
            work.hessenberg[second] = Complex64::new(0.0, 0.0);
            let g = work.rotated_rhs[inner];
            work.rotated_rhs[inner] = c * g;
            work.rotated_rhs[inner + 1] = -s.conj() * g;
            if !candidate_from_basis::<T, A, O>(
                work,
                inner + 1,
                restart,
                n,
                correction_scale,
                operations,
            )
            .await?
            {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            }
            operations.coefficient(work.candidate, work.product).await?;
            for (index, (destination, (&b, &ax))) in work
                .residual
                .iter_mut()
                .zip(rhs.iter().zip(&*work.product))
                .enumerate()
            {
                if index % 1024 == 0 {
                    operations.cooperate((n - index).min(1024) as u64).await?;
                }
                let Some(value) = A::from_complex(A::to_complex(b) - A::to_complex(ax)) else {
                    return Ok(GmresOutcome {
                        flag: 4,
                        relres: best_residual.relative(denominator),
                        outer: best_iteration.0,
                        inner: best_iteration.1,
                        residual_count,
                    });
                };
                *destination = value;
            }
            if operations
                .precondition(work.residual, work.preconditioned, work.intermediate)
                .await?
                == PreconditionerStatus::NumericalFailure
            {
                return Ok(GmresOutcome {
                    flag: 2,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            }
            let Some(actual) = norm::<T, A, O>(work.preconditioned, operations).await? else {
                return Ok(GmresOutcome {
                    flag: 4,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            };
            work.history[residual_count] = actual.magnitude();
            residual_count += 1;
            let last = (outer + 1, inner + 1);
            if actual.less_than(best_residual) {
                best_residual = actual;
                best_iteration = last;
                copy(work.candidate, work.best, operations).await?;
            }
            if actual.relative(denominator) <= tolerance {
                copy(work.candidate, work.x, operations).await?;
                return Ok(GmresOutcome {
                    flag: 0,
                    relres: actual.relative(denominator),
                    outer: outer + 1,
                    inner: inner + 1,
                    residual_count,
                });
            }
            let mut same = previous_candidate;
            for (candidate, previous) in work.candidate.chunks(1024).zip(work.previous.chunks(1024))
            {
                operations.cooperate(candidate.len() as u64).await?;
                same &= candidate == previous;
            }
            if same {
                return Ok(GmresOutcome {
                    flag: 3,
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
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
                    relres: best_residual.relative(denominator),
                    outer: best_iteration.0,
                    inner: best_iteration.1,
                    residual_count,
                });
            }
            if inner + 1 == restart {
                copy(work.candidate, work.x, operations).await?;
            }
        }
    }
    copy(work.best, work.x, operations).await?;
    Ok(GmresOutcome {
        flag: 1,
        relres: best_residual.relative(denominator),
        outer: best_iteration.0,
        inner: best_iteration.1,
        residual_count,
    })
}
