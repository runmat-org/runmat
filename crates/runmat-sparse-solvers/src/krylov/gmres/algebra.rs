use super::scalar::{copy, fill, scaled_axpy, Norm};
use super::Arithmetic;
use super::{h_index, GmresOperations, GmresWork, PreconditionerSide, PreconditionerStatus};
use crate::krylov::usage::{BatchCompletion, ScalarWork, VectorWork};

/// Result of forming the candidate iterate from the projected solution.
pub(super) enum Candidate {
    Ready,
    Breakdown,
    PreconditionerFailure,
}

pub(super) struct BasisProjection {
    pub completed: usize,
    pub restart: usize,
    pub dimension: usize,
}
use num_complex::Complex64;
pub fn rotation(a: Complex64, b: Complex64) -> Option<(f64, Complex64)> {
    let aa = if a.im == 0.0 {
        a.re.abs()
    } else {
        a.re.hypot(a.im)
    };
    let bb = if b.im == 0.0 {
        b.re.abs()
    } else {
        b.re.hypot(b.im)
    };
    if !aa.is_finite() || !bb.is_finite() {
        return None;
    }
    let size = aa.hypot(bb);
    if !size.is_finite() {
        return None;
    }
    if size == 0.0 {
        return Some((1.0, Complex64::new(0.0, 0.0)));
    }
    if aa == 0.0 {
        return Some((0.0, b.conj() / size));
    }
    Some((aa / size, (a / aa) * b.conj() / size))
}

pub(super) async fn apply_previous_rotations<T, O: GmresOperations<T>>(
    work: &mut GmresWork<'_, T>,
    column: usize,
    restart: usize,
    operations: &mut O,
) -> Result<(), O::Error> {
    for row in 0..column {
        if row % 1024 == 0 {
            operations
                .cooperate((column - row).min(1024) as u64)
                .await?;
        }
        operations
            .measured(VectorWork::Scalar(ScalarWork::ComplexRotationPair), |_| {
                let first = h_index(row, column, restart);
                let second = first + 1;
                let a = work.hessenberg[first];
                let b = work.hessenberg[second];
                let c = work.cosines[row];
                let s = work.sines[row];
                work.hessenberg[first] = c * a + s * b;
                work.hessenberg[second] = -s.conj() * a + c * b;
                ((), BatchCompletion::Complete)
            })
            .await?;
    }
    Ok(())
}

pub(super) async fn candidate_from_basis<
    T: Copy + PartialEq,
    A: Arithmetic<T>,
    O: GmresOperations<T>,
>(
    work: &mut GmresWork<'_, T>,
    projection: BasisProjection,
    beta: Norm,
    side: PreconditionerSide,
    flexible_basis: &[T],
    operations: &mut O,
) -> Result<Candidate, O::Error> {
    let BasisProjection {
        completed,
        restart,
        dimension: n,
    } = projection;
    for (source, target) in work.rotated_rhs[..completed]
        .chunks(1024)
        .zip(work.coefficients[..completed].chunks_mut(1024))
    {
        operations.cooperate(source.len() as u64).await?;
        target.copy_from_slice(source);
    }
    for row in (0..completed).rev() {
        let mut value = work.coefficients[row];
        operations.cooperate(1).await?;
        for column in row + 1..completed {
            if (column - row - 1) % 1024 == 0 {
                operations
                    .cooperate((completed - column).min(1024) as u64)
                    .await?;
            }
            value = operations
                .measured(
                    VectorWork::Scalar(ScalarWork::ComplexBacksolveUpdate),
                    |_| {
                        (
                            value
                                - work.hessenberg[h_index(row, column, restart)]
                                    * work.coefficients[column],
                            BatchCompletion::Complete,
                        )
                    },
                )
                .await?;
        }
        let diagonal = work.hessenberg[h_index(row, row, restart)];
        let diagonal_zero = operations
            .measured(
                VectorWork::Scalar(if diagonal.im == 0.0 {
                    ScalarWork::Zero
                } else {
                    ScalarWork::Hypot
                }),
                |_| {
                    (
                        if diagonal.im == 0.0 {
                            diagonal.re.abs() == 0.0
                        } else {
                            diagonal.norm() == 0.0
                        },
                        BatchCompletion::Complete,
                    )
                },
            )
            .await?;
        if diagonal_zero {
            return Ok(Candidate::Breakdown);
        }
        let (real, imaginary) = operations
            .measured(VectorWork::Scalar(ScalarWork::ComplexDivide), |_| {
                (
                    A::divide_complex((value.re, value.im), (diagonal.re, diagonal.im)),
                    BatchCompletion::Complete,
                )
            })
            .await?;
        let result = Complex64::new(real, imaginary);
        if !result.re.is_finite() || !result.im.is_finite() {
            return Ok(Candidate::Breakdown);
        }
        work.coefficients[row] = result;
    }
    match side {
        PreconditionerSide::Left => {
            // x + V y.
            copy(work.x, work.candidate, operations).await?;
            for column in 0..completed {
                let basis = &work.basis[column * n..(column + 1) * n];
                if !scaled_axpy::<T, A, O>(
                    work.candidate,
                    work.coefficients[column],
                    basis,
                    beta,
                    operations,
                )
                .await?
                {
                    return Ok(Candidate::Breakdown);
                }
            }
        }
        PreconditionerSide::Right => {
            // x + M⁻¹ (V y).
            fill(work.candidate, A::zero(), operations).await?;
            for column in 0..completed {
                let basis = &work.basis[column * n..(column + 1) * n];
                if !scaled_axpy::<T, A, O>(
                    work.candidate,
                    work.coefficients[column],
                    basis,
                    beta,
                    operations,
                )
                .await?
                {
                    return Ok(Candidate::Breakdown);
                }
            }
            if operations
                .precondition(work.candidate, work.product, work.intermediate)
                .await?
                == PreconditionerStatus::NumericalFailure
            {
                return Ok(Candidate::PreconditionerFailure);
            }
            copy(work.x, work.candidate, operations).await?;
            if !scaled_axpy::<T, A, O>(
                work.candidate,
                Complex64::new(1.0, 0.0),
                work.product,
                Norm::one(),
                operations,
            )
            .await?
            {
                return Ok(Candidate::Breakdown);
            }
        }
        PreconditionerSide::Flexible => {
            // x + Z y.
            copy(work.x, work.candidate, operations).await?;
            for column in 0..completed {
                let direction = &flexible_basis[column * n..(column + 1) * n];
                if !scaled_axpy::<T, A, O>(
                    work.candidate,
                    work.coefficients[column],
                    direction,
                    beta,
                    operations,
                )
                .await?
                {
                    return Ok(Candidate::Breakdown);
                }
            }
        }
    }
    Ok(Candidate::Ready)
}
