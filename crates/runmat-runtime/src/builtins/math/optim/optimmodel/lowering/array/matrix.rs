//! Sparse-aware affine matrix multiplication. Constant rows are indexed by
//! actual nonzeros; no Gram matrix, inverse, dense bound rows or global basis.
use super::super::sparse_rows::SparseRows;
use super::super::terms::Term;
use super::*;
use runmat_value::{NumericScalar, SparseTensor};
fn dimensions(shape: &[usize], policy: Policy) -> BuiltinResult<(usize, usize)> {
    if shape.iter().skip(2).any(|&dimension| dimension != 1) {
        return Err(policy.invalid("matrix multiplication requires two-dimensional operands"));
    }
    Ok((
        shape.first().copied().unwrap_or(1),
        shape.get(1).copied().unwrap_or(1),
    ))
}
fn sparse_value(matrix: &SparseTensor, index: usize, policy: Policy) -> BuiltinResult<f64> {
    match matrix.numeric_value_at(index) {
        Some(NumericScalar::F64(value)) => Ok(value),
        _ => Err(policy.invalid("constant sparse matrix lost its original real double storage")),
    }
}
fn emit(
    entries: &mut Vec<Term>,
    column: usize,
    row: usize,
    value: f64,
    policy: Policy,
) -> BuiltinResult<()> {
    if value != 0. {
        entries
            .try_reserve(1)
            .map_err(|source| policy.reserve(source))?;
        entries.push(Term { column, row, value });
    }
    Ok(())
}
pub(super) async fn multiply(
    left: Array,
    right: Array,
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> Lowered<Array> {
    let (m, k) = dimensions(&left.shape, policy)?;
    let (inner, n) = dimensions(&right.shape, policy)?;
    if k != inner
        || evaluation::element_count(&shape, policy)?
            != m.checked_mul(n)
                .ok_or_else(|| policy.invalid("matrix result dimensions overflow usize"))?
    {
        return Err(policy.invalid("compiled matrix dimensions differ from the original node"));
    }
    if k == 0 {
        return Ok(Ok(Array::zero(shape)));
    }
    let count = evaluation::element_count(&shape, policy)?;
    let mut left_axes = std::collections::HashSet::new();
    let mut left_rows = std::collections::HashSet::new();
    let mut right_columns = std::collections::HashSet::new();
    for source in left.degree.rows(left.count(policy)?) {
        if left.degree.at(source) {
            left_axes.try_reserve(1).map_err(|e| policy.reserve(e))?;
            left_rows.try_reserve(1).map_err(|e| policy.reserve(e))?;
            left_axes.insert(source / m);
            left_rows.insert(source % m);
        }
        cooperation.completed(1).await?;
    }
    for source in right.degree.rows(right.count(policy)?) {
        if right.degree.at(source) {
            if left_axes.contains(&(source % k)) {
                return Ok(Err(Stop::NonAffine));
            }
            right_columns
                .try_reserve(1)
                .map_err(|e| policy.reserve(e))?;
            right_columns.insert(source / k);
        }
        cooperation.completed(1).await?;
    }
    let degree = if count == 0 || (left_rows.is_empty() && right_columns.is_empty()) {
        Degree::Constant
    } else if left_rows.len() == m || right_columns.len() == n {
        Degree::Affine
    } else {
        Degree::collect(
            count,
            |row| Ok(left_rows.contains(&(row % m)) || right_columns.contains(&(row / m))),
            policy,
            cooperation,
        )
        .await?
    };
    let offset = if matches!(left.offset,Offset::Uniform(value) if value==0.)
        || matches!(right.offset,Offset::Uniform(value) if value==0.)
    {
        Offset::Uniform(0.)
    } else if let (Offset::Uniform(a), Offset::Uniform(b)) = (&left.offset, &right.offset) {
        let mut value = 0.;
        for _ in 0..k {
            value += a * b;
            if !value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            cooperation.completed(1).await?;
        }
        Offset::Uniform(value)
    } else if !matches!(left.offset, Offset::Uniform(_))
        && !matches!(right.offset, Offset::Uniform(_))
    {
        let value = evaluation::invoke(
            "mtimes",
            &[left.offset.value(), right.offset.value()],
            policy,
            cooperation,
        )
        .await?;
        if !construct::finite_numeric(value.value(), policy, cooperation).await? {
            return Ok(Err(Stop::NonFinite));
        }
        match Offset::from_value(value.value(), policy)? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        }
    } else {
        let mut values =
            runmat_value::exact_capacity_vec(count).map_err(|source| policy.storage(source))?;
        for column in 0..n {
            for row in 0..m {
                let mut value = 0.;
                for inner in 0..k {
                    value += left.offset.at(row + inner * m, policy)?
                        * right.offset.at(inner + column * k, policy)?;
                    if !value.is_finite() {
                        return Ok(Err(Stop::NonFinite));
                    }
                    cooperation.completed(1).await?;
                }
                values.push(value);
            }
        }
        Offset::Dense(
            Tensor::from_numeric_storage(runmat_value::NumericStorage::F64(values), shape.clone())
                .map_err(|source| policy.invalid(source))?,
        )
    };
    // A matrix product depends on every input participating in its dot
    // products. Missing expression slots are not numerical zero coefficients.
    let assigned =
        if matches!(left.assigned, Assigned::All) && matches!(right.assigned, Assigned::All) {
            Assigned::All
        } else {
            let mut mask =
                runmat_value::exact_capacity_vec(count).map_err(|source| policy.storage(source))?;
            let mut missing = false;
            for column in 0..n {
                for row in 0..m {
                    let mut valid = true;
                    for inner in 0..k {
                        valid &= left.assigned.at(row + inner * m, policy)?
                            && right.assigned.at(inner + column * k, policy)?;
                        cooperation.completed(1).await?;
                    }
                    mask.push(valid);
                    missing |= !valid;
                }
            }
            if missing {
                Assigned::Partial(std::rc::Rc::new(mask))
            } else {
                Assigned::All
            }
        };
    let mut entries = Vec::new();
    if !right.coefficients.is_zero() {
        for term in right.coefficients.entries() {
            let inner = term.row % k;
            let column = term.row / k;
            match &left.offset {
                Offset::Sparse(matrix) => {
                    for entry in matrix.col_ptrs[inner]..matrix.col_ptrs[inner + 1] {
                        emit(
                            &mut entries,
                            term.column,
                            matrix.row_indices[entry] + column * m,
                            term.value * sparse_value(matrix, entry, policy)?,
                            policy,
                        )?;
                        cooperation.completed(1).await?;
                    }
                }
                _ => {
                    for row in 0..m {
                        emit(
                            &mut entries,
                            term.column,
                            row + column * m,
                            term.value * left.offset.at(row + inner * m, policy)?,
                            policy,
                        )?;
                        cooperation.completed(1).await?;
                    }
                }
            }
        }
    }
    if !left.coefficients.is_zero() {
        let sparse = if let Offset::Sparse(matrix) = &right.offset {
            Some(SparseRows::prepare(matrix, policy, cooperation).await?)
        } else {
            None
        };
        for term in left.coefficients.entries() {
            let row = term.row % m;
            let inner = term.row / m;
            match (&right.offset, &sparse) {
                (Offset::Sparse(matrix), Some(index)) => {
                    for entry in index.row(inner) {
                        emit(
                            &mut entries,
                            term.column,
                            row + entry.column * m,
                            term.value * sparse_value(matrix, entry.entry, policy)?,
                            policy,
                        )?;
                        cooperation.completed(1).await?;
                    }
                }
                _ => {
                    for column in 0..n {
                        emit(
                            &mut entries,
                            term.column,
                            row + column * m,
                            term.value * right.offset.at(inner + column * k, policy)?,
                            policy,
                        )?;
                        cooperation.completed(1).await?;
                    }
                }
            }
        }
    }
    let coefficients =
        match Coefficients::from_entries(entries, count, columns, policy, cooperation).await? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        };
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        assigned,
        degree,
    }))
}
