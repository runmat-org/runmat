use super::scalar::{copy, scaled_axpy, Norm};
use super::Arithmetic;
use super::{h_index, GmresOperations, GmresWork};
use num_complex::Complex64;
pub fn rotation(a: Complex64, b: Complex64) -> Option<(f64, Complex64)> {
    let aa = a.norm();
    let bb = b.norm();
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
        let first = h_index(row, column, restart);
        let second = first + 1;
        let a = work.hessenberg[first];
        let b = work.hessenberg[second];
        let c = work.cosines[row];
        let s = work.sines[row];
        work.hessenberg[first] = c * a + s * b;
        work.hessenberg[second] = -s.conj() * a + c * b;
    }
    Ok(())
}

pub(super) async fn candidate_from_basis<
    T: Copy + PartialEq,
    A: Arithmetic<T>,
    O: GmresOperations<T>,
>(
    work: &mut GmresWork<'_, T>,
    completed: usize,
    restart: usize,
    n: usize,
    beta: Norm,
    operations: &mut O,
) -> Result<bool, O::Error> {
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
            value -= work.hessenberg[h_index(row, column, restart)] * work.coefficients[column];
        }
        let diagonal = work.hessenberg[h_index(row, row, restart)];
        if diagonal.norm() == 0.0 {
            return Ok(false);
        }
        let (real, imaginary) = A::divide_complex((value.re, value.im), (diagonal.re, diagonal.im));
        let result = Complex64::new(real, imaginary);
        if !result.re.is_finite() || !result.im.is_finite() {
            return Ok(false);
        }
        work.coefficients[row] = result;
    }
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
            return Ok(false);
        }
    }
    Ok(true)
}
