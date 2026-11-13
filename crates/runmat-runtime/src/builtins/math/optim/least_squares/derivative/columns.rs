//! Borrowed physical columns used by selected-band and error-bound arithmetic.
use super::*;
enum Column<'a> {
    Strided32 {
        values: &'a [f32],
        next: usize,
        stride: usize,
        remaining: usize,
    },
    Strided64 {
        values: &'a [f64],
        next: usize,
        stride: usize,
        remaining: usize,
    },
    Sparse32 {
        rows: &'a [usize],
        values: &'a [f32],
    },
    Sparse64 {
        rows: &'a [usize],
        values: &'a [f64],
    },
    Scalar(Option<f64>),
}
impl Iterator for Column<'_> {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Strided32 {
                values,
                next,
                stride,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                let row = *next / *stride;
                let value = values[*next];
                *remaining -= 1;
                if *remaining > 0 {
                    *next += *stride;
                }
                Some((row, value as f64))
            }
            Self::Strided64 {
                values,
                next,
                stride,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                let row = *next / *stride;
                let value = values[*next];
                *remaining -= 1;
                if *remaining > 0 {
                    *next += *stride;
                }
                Some((row, value))
            }
            Self::Sparse32 { rows, values } => {
                let (&row, rest) = rows.split_first()?;
                let (&value, remaining) = values.split_first()?;
                *rows = rest;
                *values = remaining;
                Some((row, value as f64))
            }
            Self::Sparse64 { rows, values } => {
                let (&row, rest) = rows.split_first()?;
                let (&value, remaining) = values.split_first()?;
                *rows = rest;
                *values = remaining;
                Some((row, value))
            }
            Self::Scalar(value) => value.take().map(|value| (0, value)),
        }
    }
}
impl Products<'_> {
    fn column(&self, axis: usize) -> BuiltinResult<Column<'_>> {
        use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator as V;
        let (rows, columns) = self.dimensions();
        if axis >= columns {
            return Err(invalid("selected Jacobian column is out of range"));
        }
        if rows == 0 {
            return Ok(Column::Scalar(None));
        }
        Ok(match self {
            Self::RowMajor { values, .. } => Column::Strided64 {
                values: &values[axis..],
                next: 0,
                stride: columns,
                remaining: rows,
            },
            Self::Matrix(LinearOperator::RealF32(V::Scalar(value))) => {
                Column::Scalar(Some(*value as f64))
            }
            Self::Matrix(LinearOperator::RealF64(V::Scalar(value))) => Column::Scalar(Some(*value)),
            Self::Matrix(LinearOperator::RealF32(V::Dense(view))) => Column::Strided32 {
                values: &view.as_slice()[axis * rows..][..rows],
                next: 0,
                stride: 1,
                remaining: rows,
            },
            Self::Matrix(LinearOperator::RealF64(V::Dense(view))) => Column::Strided64 {
                values: &view.as_slice()[axis * rows..][..rows],
                next: 0,
                stride: 1,
                remaining: rows,
            },
            Self::Matrix(LinearOperator::RealF32(V::Sparse(view))) => {
                let (rows, values) = view.column(axis);
                Column::Sparse32 { rows, values }
            }
            Self::Matrix(LinearOperator::RealF64(V::Sparse(view))) => {
                let (rows, values) = view.column(axis);
                Column::Sparse64 { rows, values }
            }
            _ => return Err(invalid("accepted Jacobian changed numeric class")),
        })
    }
    /// Merge only the two original selected columns, not full basis products.
    pub(in crate::builtins::math::optim::least_squares) async fn inner_product(
        &self,
        left: usize,
        right: usize,
        left_scale: f64,
        right_scale: f64,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<f64> {
        let mut a = self.column(left)?.peekable();
        let mut b = self.column(right)?.peekable();
        let mut sum = 0.;
        while let (Some(&(ar, av)), Some(&(br, bv))) = (a.peek(), b.peek()) {
            match ar.cmp(&br) {
                std::cmp::Ordering::Less => {
                    a.next();
                }
                std::cmp::Ordering::Greater => {
                    b.next();
                }
                std::cmp::Ordering::Equal => {
                    sum = (av * left_scale).mul_add(bv * right_scale, sum);
                    a.next();
                    b.next();
                }
            }
            c.completed(1).await?;
        }
        Ok(sum)
    }
    /// Formation bound |J| |input| into caller-owned retired residual scratch.
    pub(in crate::builtins::math::optim::least_squares) async fn absolute_forward(
        &self,
        input: &[f64],
        output: &mut [f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let (rows, columns) = self.dimensions();
        if input.len() != columns || output.len() != rows {
            return Err(invalid("Jacobian absolute product geometry mismatch"));
        }
        for value in output.iter_mut() {
            *value = 0.;
            c.completed(1).await?;
        }
        for (axis, &x) in input.iter().enumerate() {
            for (row, value) in self.column(axis)? {
                output[row] = value.abs().mul_add(x.abs(), output[row]);
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
        Ok(())
    }
    /// The two absolute transpose components share a single physical scan.
    pub(in crate::builtins::math::optim::least_squares) async fn absolute_components(
        &self,
        axis: usize,
        residual: &[f64],
        formed: &[f64],
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<(f64, f64)> {
        let (rows, _) = self.dimensions();
        if residual.len() != rows || formed.len() != rows {
            return Err(invalid("Jacobian stationarity geometry mismatch"));
        }
        let mut rhs = 0.;
        let mut curvature = 0.;
        for (row, value) in self.column(axis)? {
            rhs = value.abs().mul_add(residual[row].abs(), rhs);
            curvature = value.abs().mul_add(formed[row], curvature);
            c.completed(1).await?;
        }
        Ok((rhs, curvature))
    }
}
