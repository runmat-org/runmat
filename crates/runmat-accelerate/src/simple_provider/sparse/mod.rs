use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    validate_matrix_owner, validate_pattern_owner, validate_preconditioner_owner,
    validate_sparse_values, GpuTensorStorage, HostCsrPatternView, HostNumericDataView,
    HostNumericTensorView, NumericElementType, SparseMatrixDescriptor, SparseMatrixHandle,
    SparsePatternDescriptor, SparsePatternHandle, SparsePreconditionerDescriptor,
    SparsePreconditionerHandle, SparsePreconditionerKind,
};

mod algebra;
mod cancellation;
mod cg;
mod contract;
mod provider;

#[cfg(test)]
mod tests;

use algebra::{apply_jacobi, multiply};
use cancellation::{CancellationMonitor, Controlled};
pub(in crate::simple_provider) use cg::{solve_cg, CgProblem};
pub(super) use contract::{capabilities, cost, feasibility};
pub(super) use provider::{apply_preconditioner, krylov_solve, prepare_preconditioner, spmv};

#[derive(Clone)]
pub(super) struct CsrPattern {
    pub descriptor: SparsePatternDescriptor,
    pub row_offsets: Vec<u64>,
    pub column_indices: Vec<u64>,
}

#[derive(Clone)]
pub(super) struct SparseMatrix {
    pub pattern: Arc<CsrPattern>,
    pub descriptor: SparseMatrixDescriptor,
    pub values: Vec<f64>,
}

#[derive(Clone)]
pub(super) struct SparsePreconditioner {
    pub descriptor: SparsePreconditionerDescriptor,
    pub inverse_diagonal: Vec<f64>,
}

#[derive(Default)]
struct SparseStorage {
    patterns: HashMap<u64, Arc<CsrPattern>>,
    matrices: HashMap<u64, Arc<SparseMatrix>>,
    preconditioners: HashMap<u64, Arc<SparsePreconditioner>>,
}

#[derive(Default)]
pub(super) struct SparseState {
    storage: Mutex<SparseStorage>,
}

impl SparseState {
    fn lock(&self) -> MutexGuard<'_, SparseStorage> {
        self.storage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    pub fn live_bytes(&self) -> u64 {
        let storage = self.lock();
        let mut patterns = HashSet::new();
        let mut bytes = 0_u64;
        for pattern in storage
            .patterns
            .values()
            .chain(storage.matrices.values().map(|matrix| &matrix.pattern))
        {
            if patterns.insert(Arc::as_ptr(pattern) as usize) {
                bytes = bytes.saturating_add(
                    u64::try_from(
                        pattern
                            .row_offsets
                            .len()
                            .saturating_add(pattern.column_indices.len()),
                    )
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8),
                );
            }
        }
        for matrix in storage.matrices.values() {
            bytes = bytes.saturating_add(
                u64::try_from(matrix.values.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8),
            );
        }
        for preconditioner in storage.preconditioners.values() {
            bytes = bytes.saturating_add(
                u64::try_from(preconditioner.inverse_diagonal.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8),
            );
        }
        bytes
    }

    pub fn upload_pattern(
        &self,
        device_id: u32,
        pattern_id: u64,
        view: HostCsrPatternView<'_>,
    ) -> anyhow::Result<SparsePatternHandle> {
        let descriptor = view.validate()?;
        let pattern = Arc::new(CsrPattern {
            descriptor,
            row_offsets: view.row_offsets.to_vec(),
            column_indices: view.column_indices.to_vec(),
        });
        ensure!(
            self.lock().patterns.insert(pattern_id, pattern).is_none(),
            "sparse pattern identifier already exists"
        );
        Ok(SparsePatternHandle {
            device_id,
            pattern_id,
            descriptor,
        })
    }

    pub fn bind_values(
        &self,
        device_id: u32,
        matrix_id: u64,
        pattern: &SparsePatternHandle,
        values: HostNumericTensorView<'_>,
    ) -> anyhow::Result<SparseMatrixHandle> {
        validate_pattern_owner(pattern, device_id)?;
        let descriptor = validate_sparse_values(pattern, values)?;
        ensure!(
            descriptor.element_type == NumericElementType::F64
                && descriptor.storage == GpuTensorStorage::Real,
            "in-process sparse provider supports real f64 values only"
        );
        let HostNumericDataView::F64(values) = values.data else {
            return Err(anyhow!("in-process sparse provider requires f64 values"));
        };
        let mut storage = self.lock();
        let resident_pattern = storage
            .patterns
            .get(&pattern.pattern_id)
            .cloned()
            .ok_or_else(|| anyhow!("sparse pattern is not resident"))?;
        ensure!(
            resident_pattern.descriptor == pattern.descriptor,
            "sparse pattern handle descriptor does not match resident storage"
        );
        let matrix = Arc::new(SparseMatrix {
            pattern: resident_pattern,
            descriptor,
            values: values.to_vec(),
        });
        ensure!(
            storage.matrices.insert(matrix_id, matrix).is_none(),
            "sparse matrix identifier already exists"
        );
        Ok(SparseMatrixHandle {
            device_id,
            matrix_id,
            pattern: pattern.descriptor,
            descriptor,
        })
    }

    pub fn matrix(
        &self,
        device_id: u32,
        handle: &SparseMatrixHandle,
    ) -> anyhow::Result<Arc<SparseMatrix>> {
        validate_matrix_owner(handle, device_id)?;
        let matrix = self
            .lock()
            .matrices
            .get(&handle.matrix_id)
            .cloned()
            .ok_or_else(|| anyhow!("sparse matrix is not resident"))?;
        ensure!(
            matrix.pattern.descriptor == handle.pattern && matrix.descriptor == handle.descriptor,
            "sparse matrix handle does not match resident storage"
        );
        Ok(matrix)
    }

    pub fn prepare_jacobi(
        &self,
        device_id: u32,
        preconditioner_id: u64,
        matrix_handle: &SparseMatrixHandle,
    ) -> anyhow::Result<SparsePreconditionerHandle> {
        let matrix = self.matrix(device_id, matrix_handle)?;
        ensure!(
            matrix.pattern.descriptor.rows == matrix.pattern.descriptor.columns,
            "Jacobi preconditioner requires a square matrix"
        );
        let mut inverse_diagonal = Vec::with_capacity(matrix.pattern.descriptor.rows as usize);
        for row in 0..matrix.pattern.descriptor.rows as usize {
            let start = matrix.pattern.row_offsets[row] as usize;
            let end = matrix.pattern.row_offsets[row + 1] as usize;
            let diagonal = matrix.pattern.column_indices[start..end]
                .iter()
                .position(|column| *column == row as u64)
                .map(|offset| matrix.values[start + offset])
                .ok_or_else(|| anyhow!("Jacobi preconditioner requires every diagonal entry"))?;
            ensure!(
                diagonal.is_finite() && diagonal != 0.0,
                "Jacobi preconditioner requires finite nonzero diagonal entries"
            );
            inverse_diagonal.push(1.0 / diagonal);
        }
        let descriptor = SparsePreconditionerDescriptor {
            matrix_id: matrix_handle.matrix_id,
            kind: SparsePreconditionerKind::Jacobi,
            dimension: matrix.pattern.descriptor.rows,
            element_type: matrix.descriptor.element_type,
            storage: matrix.descriptor.storage,
        };
        let preconditioner = Arc::new(SparsePreconditioner {
            descriptor,
            inverse_diagonal,
        });
        ensure!(
            self.lock()
                .preconditioners
                .insert(preconditioner_id, preconditioner)
                .is_none(),
            "sparse preconditioner identifier already exists"
        );
        Ok(SparsePreconditionerHandle {
            device_id,
            preconditioner_id,
            descriptor,
        })
    }

    pub fn preconditioner(
        &self,
        device_id: u32,
        handle: &SparsePreconditionerHandle,
    ) -> anyhow::Result<Arc<SparsePreconditioner>> {
        validate_preconditioner_owner(handle, device_id)?;
        let preconditioner = self
            .lock()
            .preconditioners
            .get(&handle.preconditioner_id)
            .cloned()
            .ok_or_else(|| anyhow!("sparse preconditioner is not resident"))?;
        ensure!(
            preconditioner.descriptor == handle.descriptor,
            "sparse preconditioner handle does not match resident storage"
        );
        Ok(preconditioner)
    }

    pub fn free_pattern(&self, device_id: u32, handle: &SparsePatternHandle) -> anyhow::Result<()> {
        validate_pattern_owner(handle, device_id)?;
        ensure!(
            self.lock().patterns.remove(&handle.pattern_id).is_some(),
            "sparse pattern is not resident"
        );
        Ok(())
    }

    pub fn free_matrix(&self, device_id: u32, handle: &SparseMatrixHandle) -> anyhow::Result<()> {
        validate_matrix_owner(handle, device_id)?;
        let mut storage = self.lock();
        ensure!(
            storage.matrices.remove(&handle.matrix_id).is_some(),
            "sparse matrix is not resident"
        );
        storage
            .preconditioners
            .retain(|_, value| value.descriptor.matrix_id != handle.matrix_id);
        Ok(())
    }

    pub fn free_preconditioner(
        &self,
        device_id: u32,
        handle: &SparsePreconditionerHandle,
    ) -> anyhow::Result<()> {
        validate_preconditioner_owner(handle, device_id)?;
        ensure!(
            self.lock()
                .preconditioners
                .remove(&handle.preconditioner_id)
                .is_some(),
            "sparse preconditioner is not resident"
        );
        Ok(())
    }
}

pub(super) fn read_f64_vector(
    device_id: u32,
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> anyhow::Result<Vec<f64>> {
    ensure!(
        handle.device_id() == device_id,
        "sparse vector belongs to a different provider"
    );
    ensure!(
        handle.descriptor().element_type == Some(NumericElementType::F64)
            && handle.descriptor().storage == Some(GpuTensorStorage::Real),
        "in-process sparse provider requires real f64 vectors"
    );
    let registry = super::numeric_registry()
        .lock_for(device_id)
        .map_err(|_| anyhow!("sparse vector allocation owner is poisoned"))?;
    match registry.get(handle) {
        Some(super::InProcessNumericData::F64(values)) => Ok(values.clone()),
        Some(_) => Err(anyhow!(
            "sparse vector storage does not match its descriptor"
        )),
        None => Err(anyhow!("sparse vector is not resident")),
    }
}
