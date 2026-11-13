//! Sparse chain rule for original vector-objective identities. The exact
//! attainment column never consumes a user callback or finite-difference point.
use super::formulation::{capacity, Formulation};
use super::{invalid, Kind};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;

impl Formulation {
    /// Recover the public n-by-m objective Jacobian from the first typed row
    /// for each objective. This is an observable projection, not a dense solver
    /// fallback or a second retained numerical copy.
    pub(crate) async fn original_jacobian(
        &self,
        source: &CscMatrix<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<CscMatrix<f64>> {
        let n = source
            .n
            .checked_sub(1)
            .ok_or_else(|| invalid(self.kind, "missing attainment column"))?;
        if source.m < self.rows.len() {
            return Err(invalid(
                self.kind,
                "objective row projection is inconsistent",
            ));
        }
        let mut offsets = capacity(self.objectives + 1, self.kind)?;
        for _ in 0..=self.objectives {
            offsets.push(0usize);
            cooperation.completed(1).await?;
        }
        for column in 0..n {
            for &row in &source.rowval[source.colptr[column]..source.colptr[column + 1]] {
                if let Ok(objective) = self.first_row[..self.objectives].binary_search(&row) {
                    offsets[objective + 1] += 1;
                }
                cooperation.completed(1).await?;
            }
        }
        for column in 0..self.objectives {
            offsets[column + 1] += offsets[column];
            cooperation.completed(1).await?;
        }
        let count = *offsets.last().expect("objective column extent");
        let mut next = capacity(self.objectives, self.kind)?;
        for &offset in &offsets[..self.objectives] {
            next.push(offset);
            cooperation.completed(1).await?;
        }
        let mut rows = capacity(count, self.kind)?;
        let mut values = capacity(count, self.kind)?;
        for _ in 0..count {
            rows.push(0);
            values.push(0.0);
            cooperation.completed(1).await?;
        }
        for column in 0..n {
            for index in source.colptr[column]..source.colptr[column + 1] {
                let row = source.rowval[index];
                if let Ok(objective) = self.first_row[..self.objectives].binary_search(&row) {
                    let destination = next[objective];
                    rows[destination] = column;
                    values[destination] =
                        source.nzval[index] / self.rows[row].direction.coefficient();
                    next[objective] += 1;
                }
                cooperation.completed(1).await?;
            }
        }
        Ok(CscMatrix::new(n, self.objectives, offsets, rows, values))
    }
    /// The original public n-by-m Jacobian has already been validated and
    /// transposed by the existing callback decoder to m-by-n CSC.
    pub(crate) async fn jacobian(
        &self,
        original: CscMatrix<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<CscMatrix<f64>>> {
        if original.m != self.objectives {
            return Err(invalid(
                self.kind,
                "objective Jacobian has inconsistent row identities",
            ));
        }
        let variables = original
            .n
            .checked_add(1)
            .ok_or_else(|| invalid(self.kind, "attainment variable extent overflow"))?;
        let mut offsets = capacity(
            variables
                .checked_add(1)
                .ok_or_else(|| invalid(self.kind, "Jacobian column extent overflow"))?,
            self.kind,
        )?;
        let mut rows = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for column in 0..original.n {
            for source in original.colptr[column]..original.colptr[column + 1] {
                let objective = original.rowval[source];
                let coefficient = original.nzval[source];
                if !coefficient.is_finite() {
                    return Ok(None);
                }
                for row in self.first_row[objective]..self.first_row[objective + 1] {
                    let value = coefficient * self.rows[row].direction.coefficient();
                    if value != 0.0 {
                        push(&mut rows, row, self.kind)?;
                        push(&mut values, value, self.kind)?;
                    }
                    cooperation.completed(1).await?;
                }
            }
            offsets.push(rows.len());
            cooperation.completed(1).await?;
        }
        for (row, value) in self.rows.iter().enumerate() {
            if value.weight != 0.0 {
                push(&mut rows, row, self.kind)?;
                push(&mut values, -value.weight, self.kind)?;
            }
            cooperation.completed(1).await?;
        }
        offsets.push(rows.len());
        Ok(Some(CscMatrix::new(
            self.rows.len(),
            variables,
            offsets,
            rows,
            values,
        )))
    }
}

pub(super) fn push<T>(values: &mut Vec<T>, value: T, kind: Kind) -> BuiltinResult<()> {
    if values.len() == values.capacity() {
        values.try_reserve(1).map_err(|source| {
            crate::build_runtime_error(format!(
                "{}: epigraph Jacobian allocation failed",
                kind.name()
            ))
            .with_builtin(kind.name())
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
        })?;
    }
    values.push(value);
    Ok(())
}

/// Attach original nonlinear rows beneath generated objective rows. Existing
/// nonlinear Jacobians have no gamma column; that column is exactly zero.
pub(crate) async fn stack(
    generated: CscMatrix<f64>,
    original: CscMatrix<f64>,
    kind: Kind,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<CscMatrix<f64>> {
    if original.n.checked_add(1) != Some(generated.n) {
        return Err(invalid(
            kind,
            "nonlinear Jacobian variable extent is inconsistent",
        ));
    }
    let count = generated
        .m
        .checked_add(original.m)
        .ok_or_else(|| invalid(kind, "nonlinear row extent overflow"))?;
    let mut offsets = capacity(
        generated
            .n
            .checked_add(1)
            .ok_or_else(|| invalid(kind, "nonlinear column extent overflow"))?,
        kind,
    )?;
    let entries = generated
        .nzval
        .len()
        .checked_add(original.nzval.len())
        .ok_or_else(|| invalid(kind, "nonlinear Jacobian extent overflow"))?;
    let mut rows = capacity(entries, kind)?;
    let mut values = capacity(entries, kind)?;
    offsets.push(0);
    for column in 0..generated.n {
        for source in generated.colptr[column]..generated.colptr[column + 1] {
            rows.push(generated.rowval[source]);
            values.push(generated.nzval[source]);
            cooperation.completed(1).await?;
        }
        if column < original.n {
            for source in original.colptr[column]..original.colptr[column + 1] {
                rows.push(generated.m + original.rowval[source]);
                values.push(original.nzval[source]);
                cooperation.completed(1).await?;
            }
        }
        offsets.push(rows.len());
        cooperation.completed(1).await?;
    }
    Ok(CscMatrix::new(count, generated.n, offsets, rows, values))
}
