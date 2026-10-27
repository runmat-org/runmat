//! One mutable column-major callback argument, not a second point history.
use super::{invalid, Provenance, ScalarCooperation};
use crate::BuiltinResult;
use std::num::NonZeroUsize;
pub(super) struct Row {
    pub provenance: Provenance,
    pub seed: Option<super::Seed>,
}

pub(super) struct Batch {
    coordinates: Vec<f64>,
    provenance: Vec<Row>,
    target: NonZeroUsize,
    variables: usize,
}
impl Batch {
    pub(super) async fn new(
        target: NonZeroUsize,
        variables: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let length = target
            .get()
            .checked_mul(variables)
            .ok_or_else(|| invalid("staged callback batch extent overflow"))?;
        crate::context::host_allocation::require_unbounded_workspace(
            cooperation.runtime(),
            super::super::NAME,
            "finite host-memory execution has no complete staged callback workspace bound",
        )?;
        let mut coordinates = runmat_value::exact_capacity_vec(length).map_err(allocation)?;
        let provenance = runmat_value::exact_capacity_vec(target.get()).map_err(allocation)?;
        while coordinates.len() < length {
            let end = length.min(coordinates.len().saturating_add(1024));
            let work = end - coordinates.len();
            coordinates.resize(end, 0.0);
            cooperation.completed(work as u64).await?;
        }
        Ok(Self {
            coordinates,
            provenance,
            target,
            variables,
        })
    }
    pub(super) fn len(&self) -> usize {
        self.provenance.len()
    }
    pub(super) fn full(&self) -> bool {
        self.len() == self.target.get()
    }
    pub(super) fn planned(&self, phase: super::queue::PhaseId) -> usize {
        self.provenance
            .first()
            .filter(|first| first.provenance.phase == phase)
            .map_or(0, |_| self.len())
    }
    pub(super) async fn contains_point(
        &self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if point.len() != self.variables {
            return Ok(false);
        }
        for row in 0..self.len() {
            let mut equal = true;
            for (column, &value) in point.iter().enumerate() {
                equal &= self.coordinates[column * self.target.get() + row] == value;
                cooperation.completed(1).await?;
                if !equal {
                    break;
                }
            }
            if equal {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub(super) async fn push(
        &mut self,
        point: Vec<f64>,
        provenance: Provenance,
        seed: Option<super::Seed>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if self.full() || point.len() != self.variables {
            return Err(invalid("staged callback point changed batch geometry"));
        }
        if self
            .provenance
            .first()
            .is_some_and(|first| first.provenance.phase != provenance.phase)
        {
            return Err(invalid("staged callback batch crosses a surrogate reset"));
        }
        let row = self.len();
        for (column, value) in point.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid("staged callback point is nonfinite"));
            }
            self.coordinates[column * self.target.get() + row] = value;
            cooperation.completed(1).await?;
        }
        self.provenance.push(Row { provenance, seed });
        Ok(())
    }
    pub(super) fn into_full(self) -> BuiltinResult<(Vec<f64>, NonZeroUsize, Vec<Row>)> {
        if !self.full() {
            return Err(invalid("incomplete vectorized batch cannot be invoked"));
        }
        Ok((self.coordinates, self.target, self.provenance))
    }
    pub(super) async fn into_final(
        mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(Vec<f64>, NonZeroUsize, Vec<Row>)> {
        let rows =
            NonZeroUsize::new(self.len()).ok_or_else(|| invalid("final staged batch is empty"))?;
        if rows != self.target {
            for column in 1..self.variables {
                let source = column * self.target.get();
                let destination = column * rows.get();
                let mut copied = 0;
                while copied < rows.get() {
                    let work = (rows.get() - copied).min(1024);
                    self.coordinates.copy_within(
                        source + copied..source + copied + work,
                        destination + copied,
                    );
                    copied += work;
                    cooperation.completed(work as u64).await?;
                }
            }
            self.coordinates.truncate(rows.get() * self.variables);
        }
        Ok((self.coordinates, rows, self.provenance))
    }
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("surrogateopt: staged callback allocation failed")
        .with_builtin(super::super::NAME)
        .with_source(source)
        .build()
}
