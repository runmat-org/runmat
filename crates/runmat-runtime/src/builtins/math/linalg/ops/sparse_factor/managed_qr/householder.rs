use super::{csc::Csc, norm::Norm, SparseCompleteQrError as Error, SparseQrScalar, Work};

/// Unit Hermitian reflector. Scaling the input before forming x+phase*norm
/// avoids cancellation and overflow in the reflector itself.
pub(super) async fn append<T: SparseQrScalar>(
    source: &Csc<T>,
    column: usize,
    pivot: usize,
    norm: Norm,
    v: &mut Csc<T>,
    work: &mut Work<'_>,
) -> Result<T, Error> {
    let x = source.at(pivot, column);
    let mut head = Norm::zero();
    head.add(x);
    let phase = if head.scale == 0.0 {
        T::one()
    } else {
        head.unit(x)
    };
    let start = v.values.len();
    v.push(pivot, norm.unit(x) + phase, work).await?;
    let (rows, values) = source.column(column);
    for (&row, &value) in rows.iter().zip(values) {
        if row > pivot {
            v.push(row, norm.unit(value), work).await?;
        }
        work.tick().await?;
    }
    let mut magnitude = Norm::zero();
    for &value in &v.values.as_slice()[start..] {
        magnitude.add(value);
        work.tick().await?;
    }
    for i in start..v.values.len() {
        let value = magnitude.unit(v.values.as_slice()[i]);
        v.values.as_mut_slice()[i] = value;
        work.tick().await?;
    }
    v.end_column(pivot);
    let diagonal = phase * T::real(-norm.magnitude());
    if !diagonal.finite() {
        return Err(Error::NumericalBreakdown);
    }
    Ok(diagonal)
}

/// Merge only reflector support with the column's existing support. Fill is
/// appended through exact growth; there is no dense row accumulator.
pub(super) async fn transform<T: SparseQrScalar>(
    source: &Csc<T>,
    column: usize,
    v: &Csc<T>,
    pivot: usize,
    destination: &mut Csc<T>,
    work: &mut Work<'_>,
) -> Result<(), Error> {
    let (rows, values) = source.column(column);
    let (vr, vv) = v.column(pivot);
    let (mut a, mut b) = (0, 0);
    let mut dot = T::default();
    while a < rows.len() && b < vr.len() {
        match rows[a].cmp(&vr[b]) {
            std::cmp::Ordering::Less => a += 1,
            std::cmp::Ordering::Greater => b += 1,
            std::cmp::Ordering::Equal => {
                dot = dot + vv[b].conjugate() * values[a];
                a += 1;
                b += 1;
            }
        }
        if !dot.finite() {
            return Err(Error::NumericalBreakdown);
        }
        work.tick().await?;
    }
    a = 0;
    b = 0;
    while a < rows.len() || b < vr.len() {
        let row = match (rows.get(a), vr.get(b)) {
            (Some(&a), Some(&b)) => a.min(b),
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => unreachable!(),
        };
        let original = if rows.get(a) == Some(&row) {
            let value = values[a];
            a += 1;
            value
        } else {
            T::default()
        };
        let reflect = if vr.get(b) == Some(&row) {
            let value = vv[b];
            b += 1;
            value
        } else {
            T::default()
        };
        let delta = reflect * dot;
        destination
            .push(row, (original - delta) - delta, work)
            .await?;
    }
    Ok(())
}

pub(super) async fn apply<T: SparseQrScalar>(
    v: &Csc<T>,
    column: usize,
    rhs: &mut [T],
    work: &mut Work<'_>,
) -> Result<(), Error> {
    let (rows, values) = v.column(column);
    apply_column(rows, values, rhs, work).await
}

pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn apply_column<
    T: SparseQrScalar,
>(
    rows: &[usize],
    values: &[T],
    rhs: &mut [T],
    work: &mut Work<'_>,
) -> Result<(), Error> {
    apply_column_as::<T, T>(rows, values, rhs, work).await
}
pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn apply_column_as<
    T: SparseQrScalar,
    U: SparseQrScalar,
>(
    rows: &[usize],
    values: &[T],
    rhs: &mut [U],
    work: &mut Work<'_>,
) -> Result<(), Error> {
    let mut dot = U::default();
    for (&row, &value) in rows.iter().zip(values) {
        let value =
            crate::builtins::math::linalg::ops::sparse_factor::coefficient_as::<T, U>(value)
                .ok_or(Error::UnsupportedClass)?;
        dot = dot + value.conjugate() * rhs[row];
        if !dot.finite() {
            return Err(Error::NumericalBreakdown);
        }
        work.tick().await?;
    }
    for (&row, &value) in rows.iter().zip(values) {
        let value =
            crate::builtins::math::linalg::ops::sparse_factor::coefficient_as::<T, U>(value)
                .ok_or(Error::UnsupportedClass)?;
        let delta = value * dot;
        let result = (rhs[row] - delta) - delta;
        if !result.finite() {
            return Err(Error::NumericalBreakdown);
        }
        rhs[row] = result;
        work.tick().await?;
    }
    Ok(())
}
