//! Symmetric Hessian sparsity masks and shared greedy column coloring.
use super::*;
use runmat_value::{HostIndexBuffer, Value};
pub(crate) struct Pattern {
    pub(crate) columns: HostIndexBuffer,
    pub(crate) rows: HostIndexBuffer,
    pub(crate) colors: Vec<usize>,
    pub(crate) count: usize,
}

impl Pattern {
    pub(crate) async fn prepare(
        value: &Value,
        n: usize,
        cooperation: &mut ScalarCooperation,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        let extent = n
            .checked_add(1)
            .ok_or_else(|| policy.invalid("HessPattern extent overflow"))?;
        let mut columns = policy.buffer(extent)?;
        columns.push(0);
        let mut rows;
        match value {
            Value::SparseTensor(value)
                if value.rows == n && value.cols == n && !value.is_complex() =>
            {
                let mut proof = runmat_value::SparseCscValidation::new(
                    n,
                    n,
                    value.col_ptrs.clone(),
                    value.row_indices.clone(),
                    value.nnz(),
                )
                .map_err(|error| policy.invalid(error))?;
                let quantum =
                    std::num::NonZeroUsize::new(1024).expect("nonzero validation quantum");
                while !proof
                    .advance(quantum)
                    .map_err(|error| policy.invalid(error))?
                {
                    cooperation.completed(1024).await?;
                }
                let _ = proof
                    .into_validated()
                    .map_err(|_| policy.invalid("HessPattern validation did not complete"))?;
                rows = policy.buffer(value.nnz())?;
                for column in 0..n {
                    for index in value.col_ptrs[column]..value.col_ptrs[column + 1] {
                        let nonzero = if value.is_logical() {
                            true
                        } else {
                            let scalar = value
                                .numeric_host_buffer()
                                .and_then(|storage| storage.value_at(index))
                                .ok_or_else(|| policy.invalid("invalid HessPattern storage"))?;
                            let number = scalar.materialize_f64();
                            if !number.is_finite() {
                                return Err(policy.invalid("HessPattern must be finite"));
                            }
                            number != 0.0
                        };
                        if nonzero {
                            rows.push(value.row_indices[index]);
                        }
                        cooperation.completed(1).await?;
                    }
                    columns.push(rows.len());
                    cooperation.completed(1).await?;
                }
            }
            Value::Tensor(value) if square(value.shape(), n) => {
                rows = policy.buffer(value.len())?;
                for column in 0..n {
                    for row in 0..n {
                        let number = crate::builtins::common::tensor::tensor_value_f64(
                            value,
                            column * n + row,
                        );
                        if !number.is_finite() {
                            return Err(policy.invalid("HessPattern must be finite"));
                        }
                        if number != 0.0 {
                            rows.push(row);
                        }
                        cooperation.completed(1).await?;
                    }
                    columns.push(rows.len());
                }
            }
            Value::LogicalArray(value) if square(&value.shape, n) => {
                rows = policy.buffer(value.data.len())?;
                for column in 0..n {
                    for row in 0..n {
                        if value.data[column * n + row] != 0 {
                            rows.push(row);
                        }
                        cooperation.completed(1).await?;
                    }
                    columns.push(rows.len());
                }
            }
            Value::Num(number) if n == 1 && number.is_finite() => {
                rows = policy.buffer(1)?;
                if *number != 0.0 {
                    rows.push(0);
                }
                columns.push(rows.len());
            }
            Value::Bool(value) if n == 1 => {
                rows = policy.buffer(1)?;
                if *value {
                    rows.push(0);
                }
                columns.push(rows.len());
            }
            Value::Int(value) if n == 1 => {
                rows = policy.buffer(1)?;
                if value.try_to_u64() != Some(0) {
                    rows.push(0);
                }
                columns.push(rows.len());
            }
            _ => {
                return Err(
                    policy.invalid("HessPattern must be a finite real n-by-n structural matrix")
                )
            }
        }
        // Hessian structure must be symmetric. Numerical entries are only
        // masks; unequal nonzero weights do not create a false asymmetry.
        for column in 0..n {
            cooperation.completed(1).await?;
            for &row in &rows[columns[column]..columns[column + 1]] {
                if rows[columns[row]..columns[row + 1]]
                    .binary_search(&column)
                    .is_err()
                {
                    return Err(policy.invalid("HessPattern must have symmetric nonzero structure"));
                }
                cooperation.completed(1).await?;
            }
        }
        let mut offsets = policy.filled(extent, 0usize, cooperation).await?;
        for &row in &rows {
            offsets[row + 1] += 1;
            cooperation.completed(1).await?;
        }
        for row in 0..n {
            offsets[row + 1] += offsets[row];
            cooperation.completed(1).await?;
        }
        let mut next = policy.buffer(n)?;
        for chunk in offsets[..n].chunks(1024) {
            next.extend_from_slice(chunk);
            cooperation.completed(chunk.len() as u64).await?;
        }
        let mut adjacent = policy.filled(rows.len(), 0usize, cooperation).await?;
        for column in 0..n {
            for &row in &rows[columns[column]..columns[column + 1]] {
                adjacent[next[row]] = column;
                next[row] += 1;
                cooperation.completed(1).await?;
            }
            cooperation.completed(1).await?;
        }
        let mut colors = policy.buffer(n)?;
        let mut marks = policy.filled(n, usize::MAX, cooperation).await?;
        let mut count = 0;
        for column in 0..n {
            cooperation.completed(1).await?;
            if columns[column] == columns[column + 1] {
                colors.push(usize::MAX);
                continue;
            }
            for &row in &rows[columns[column]..columns[column + 1]] {
                for &other in &adjacent[offsets[row]..offsets[row + 1]] {
                    if other < column {
                        marks[colors[other]] = column;
                    }
                    cooperation.completed(1).await?;
                }
            }
            let color = (0..count)
                .find(|&color| marks[color] != column)
                .unwrap_or(count);
            count = count.max(color + 1);
            colors.push(color);
        }
        Ok(Self {
            columns: columns.into(),
            rows: rows.into(),
            colors,
            count,
        })
    }
}

fn square(shape: &[usize], n: usize) -> bool {
    shape.first().copied().unwrap_or(1) == n
        && shape.get(1).copied().unwrap_or(1) == n
        && shape.iter().skip(2).all(|&size| size == 1)
}
