//! Row indexing over the original borrowed CSC prefix. Only column/ordinal
//! metadata is retained; coefficients remain in the single feasibility domain.
use super::{domain::BoxView, invalid};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

pub(super) struct OriginalRows {
    offsets: Vec<usize>,
    entries: Vec<(usize, usize)>,
}
fn metadata<T>(n: usize) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(n).map_err(|source| {
        crate::build_runtime_error("surrogateopt: original row-index allocation failed")
            .with_builtin(super::NAME)
            .with_source(source)
            .build()
    })
}
impl OriginalRows {
    pub(super) async fn exact_slack(
        &self,
        domain: &BoxView<'_>,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<num_rational::BigRational>> {
        if point.len() != domain.variables() {
            return Err(invalid("original row activity has invalid point extent"));
        }
        let original = domain.linear();
        let rows = self.offsets.len() - 1;
        if rows > original.rows() {
            return Err(invalid("original activity row prefix changed"));
        }
        let mut slack = metadata(rows)?;
        for &rhs in &original.rhs()[..rows] {
            slack.push(num_rational::BigRational::from_float(rhs).expect("validated original RHS"));
            cooperation.completed(1).await?;
        }
        for (column, &value) in point.iter().enumerate() {
            let value = num_rational::BigRational::from_float(value)
                .ok_or_else(|| invalid("nonfinite original point coordinate"))?;
            for (row, coefficient) in original
                .column(column)
                .ok_or_else(|| invalid("original activity column is missing"))?
            {
                if row >= rows {
                    break;
                }
                slack[row] -= num_rational::BigRational::from_float(coefficient)
                    .expect("validated original coefficient")
                    * &value;
                cooperation.completed(1).await?;
            }
            cooperation.completed(1).await?;
        }
        Ok(slack)
    }
    pub(super) async fn new(
        domain: &BoxView<'_>,
        rows: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let original = domain.linear();
        if rows > original.rows() {
            return Err(invalid("original row prefix exceeds the admitted model"));
        }
        let extent = rows
            .checked_add(1)
            .ok_or_else(|| invalid("original row extent overflow"))?;
        let mut offsets = metadata(extent)?;
        offsets.resize(extent, 0usize);
        for &column in domain.free_columns() {
            for (row, _) in original
                .column(column)
                .ok_or_else(|| invalid("original linear column is missing"))?
            {
                if row >= rows {
                    break;
                }
                offsets[row + 1] = offsets[row + 1]
                    .checked_add(1)
                    .ok_or_else(|| invalid("original row metadata extent overflow"))?;
                cooperation.completed(1).await?;
            }
            cooperation.completed(1).await?;
        }
        for row in 0..rows {
            offsets[row + 1] = offsets[row + 1]
                .checked_add(offsets[row])
                .ok_or_else(|| invalid("original row metadata extent overflow"))?;
            cooperation.completed(1).await?;
        }
        let mut entries = metadata(offsets[rows])?;
        entries.resize(offsets[rows], (0, 0));
        let mut next = metadata(rows)?;
        next.extend_from_slice(&offsets[..rows]);
        for (axis, &column) in domain.free_columns().iter().enumerate() {
            for (ordinal, (row, _)) in original
                .column(column)
                .ok_or_else(|| invalid("original linear column is missing"))?
                .enumerate()
            {
                if row >= rows {
                    break;
                }
                entries[next[row]] = (axis, ordinal);
                next[row] += 1;
                cooperation.completed(1).await?;
            }
            cooperation.completed(1).await?;
        }
        Ok(Self { offsets, entries })
    }

    pub(super) async fn coefficients(
        &self,
        row: usize,
        domain: &BoxView<'_>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<(usize, f64)>> {
        let start = *self
            .offsets
            .get(row)
            .ok_or_else(|| invalid("original row is absent"))?;
        let end = *self
            .offsets
            .get(row + 1)
            .ok_or_else(|| invalid("original row is absent"))?;
        let original = domain.linear();
        let mut values = metadata(end - start)?;
        for &(axis, ordinal) in &self.entries[start..end] {
            let column = *domain
                .free_columns()
                .get(axis)
                .ok_or_else(|| invalid("original free coordinate changed"))?;
            let (actual_row, value) = original
                .entry(column, ordinal)
                .ok_or_else(|| invalid("original linear entry changed"))?;
            if actual_row != row {
                return Err(invalid(
                    "original row metadata changed its coefficient owner",
                ));
            }
            values.push((axis, value));
            cooperation.completed(1).await?;
        }
        Ok(values)
    }
}
