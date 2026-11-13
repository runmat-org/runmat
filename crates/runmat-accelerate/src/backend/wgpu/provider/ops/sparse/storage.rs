use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    validate_matrix_owner, validate_pattern_owner, validate_preconditioner_owner,
    validate_sparse_values, GpuTensorStorage, HostCsrPatternView, HostNumericDataView,
    HostNumericTensorView, SparseMatrixDescriptor, SparseMatrixHandle, SparsePatternDescriptor,
    SparsePatternHandle, SparsePreconditionerDescriptor, SparsePreconditionerHandle,
    WgpuBufferOwner,
};

use super::{feasibility, numeric_element_type, upload_operand};

pub(in crate::backend::wgpu::provider) struct DeviceCsrPattern {
    pub(super) descriptor: SparsePatternDescriptor,
    pub(super) row_offsets: WgpuBufferOwner,
    pub(super) column_indices: WgpuBufferOwner,
}

pub(in crate::backend::wgpu::provider) struct DeviceSparseMatrix {
    pub(super) pattern: Arc<DeviceCsrPattern>,
    pub(super) descriptor: SparseMatrixDescriptor,
    pub(super) values: WgpuBufferOwner,
}

pub(in crate::backend::wgpu::provider) struct DeviceSparsePreconditioner {
    pub(super) descriptor: SparsePreconditionerDescriptor,
    pub(super) pattern: SparsePatternDescriptor,
    pub(super) inverse_diagonal: WgpuBufferOwner,
}

#[derive(Default)]
struct SparseStorage {
    patterns: HashMap<u64, Arc<DeviceCsrPattern>>,
    matrices: HashMap<u64, Arc<DeviceSparseMatrix>>,
    preconditioners: HashMap<u64, Arc<DeviceSparsePreconditioner>>,
}

#[derive(Default)]
pub(in crate::backend::wgpu::provider) struct SparseState {
    /// Device multigrid hierarchies and their pipelines.
    pub(in crate::backend::wgpu::provider) multigrid: super::MultigridState,
    storage: Mutex<SparseStorage>,
    transient_bytes: AtomicU64,
    next_id: AtomicU64,
}

pub(super) struct SparseTransientReservation<'a> {
    state: &'a SparseState,
    bytes: u64,
}

impl Drop for SparseTransientReservation<'_> {
    fn drop(&mut self) {
        let previous = self
            .state
            .transient_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes);
    }
}

impl SparseState {
    /// Sparse handle identities are provider-local and never alias dense allocations.
    pub(in crate::backend::wgpu::provider) fn next_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    fn lock(&self) -> MutexGuard<'_, SparseStorage> {
        self.storage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    pub(super) fn reserve_transient(
        &self,
        bytes: u64,
    ) -> anyhow::Result<SparseTransientReservation<'_>> {
        self.transient_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(bytes)
            })
            .map_err(|_| anyhow!("sparse transient-byte accounting overflow"))?;
        Ok(SparseTransientReservation { state: self, bytes })
    }

    pub(in crate::backend::wgpu::provider) fn upload_pattern(
        &self,
        provider: &super::super::WgpuProvider,
        pattern_id: u64,
        view: HostCsrPatternView<'_>,
    ) -> anyhow::Result<SparsePatternHandle> {
        let descriptor = view.validate()?;
        ensure!(
            descriptor.index_width == runmat_accelerate_api::SparseIndexWidth::U32,
            "WGPU sparse kernels require u32 CSR indices"
        );
        ensure!(
            feasibility(
                provider,
                &runmat_accelerate_api::SparseFeasibilityQuery {
                    operation: runmat_accelerate_api::SparseOperation::UploadPattern,
                    pattern: descriptor,
                    element_type: numeric_element_type(provider.precision),
                    storage: GpuTensorStorage::Real,
                    vector_count: 0,
                    vector_length: 0,
                    max_iterations: None,
                },
            )
            .is_supported(),
            "sparse pattern exceeds the advertised WGPU provider envelope"
        );
        let row_offsets = view
            .row_offsets
            .iter()
            .map(|value| u32::try_from(*value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!("sparse row offset exceeds WGPU u32 indexing"))?;
        let column_indices = view
            .column_indices
            .iter()
            .map(|value| u32::try_from(*value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!("sparse column index exceeds WGPU u32 indexing"))?;
        let (row_offsets, column_indices) = provider.with_emission(|emission, _| {
            anyhow::Ok((
                upload_operand(
                    provider,
                    emission,
                    "runmat-sparse-row-offsets",
                    bytemuck::cast_slice(&row_offsets),
                    4,
                )?,
                upload_operand(
                    provider,
                    emission,
                    "runmat-sparse-column-indices",
                    bytemuck::cast_slice(&column_indices),
                    4,
                )?,
            ))
        })??;
        let pattern = Arc::new(DeviceCsrPattern {
            descriptor,
            row_offsets,
            column_indices,
        });
        ensure!(
            self.lock().patterns.insert(pattern_id, pattern).is_none(),
            "sparse pattern identifier already exists"
        );
        Ok(SparsePatternHandle {
            device_id: provider.runtime_device_id,
            pattern_id,
            descriptor,
        })
    }

    pub(in crate::backend::wgpu::provider) fn bind_values(
        &self,
        provider: &super::super::WgpuProvider,
        matrix_id: u64,
        pattern: &SparsePatternHandle,
        values: HostNumericTensorView<'_>,
    ) -> anyhow::Result<SparseMatrixHandle> {
        validate_pattern_owner(pattern, provider.runtime_device_id)?;
        let descriptor = validate_sparse_values(pattern, values)?;
        let expected_type = numeric_element_type(provider.precision);
        ensure!(
            descriptor.element_type == expected_type
                && descriptor.storage == GpuTensorStorage::Real,
            "WGPU sparse values must match the provider's real floating precision"
        );
        ensure!(
            feasibility(
                provider,
                &runmat_accelerate_api::SparseFeasibilityQuery {
                    operation: runmat_accelerate_api::SparseOperation::BindValues,
                    pattern: pattern.descriptor,
                    element_type: descriptor.element_type,
                    storage: descriptor.storage,
                    vector_count: 0,
                    vector_length: 0,
                    max_iterations: None,
                },
            )
            .is_supported(),
            "sparse values exceed the advertised WGPU provider envelope"
        );
        let bytes: &[u8] = match values.data {
            HostNumericDataView::F64(data) => bytemuck::cast_slice(data),
            HostNumericDataView::F32(data) => bytemuck::cast_slice(data),
            _ => return Err(anyhow!("WGPU sparse values require floating storage")),
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
        let values = provider.with_emission(|emission, _| {
            upload_operand(
                provider,
                emission,
                "runmat-sparse-values",
                bytes,
                descriptor.element_type.element_size() as u64,
            )
        })??;
        let matrix = Arc::new(DeviceSparseMatrix {
            pattern: resident_pattern,
            descriptor,
            values,
        });
        ensure!(
            storage.matrices.insert(matrix_id, matrix).is_none(),
            "sparse matrix identifier already exists"
        );
        Ok(SparseMatrixHandle {
            device_id: provider.runtime_device_id,
            matrix_id,
            pattern: pattern.descriptor,
            descriptor,
        })
    }

    pub(in crate::backend::wgpu::provider) fn matrix(
        &self,
        provider: &super::super::WgpuProvider,
        handle: &SparseMatrixHandle,
    ) -> anyhow::Result<Arc<DeviceSparseMatrix>> {
        validate_matrix_owner(handle, provider.runtime_device_id)?;
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

    pub(in crate::backend::wgpu::provider) fn insert_preconditioner(
        &self,
        preconditioner_id: u64,
        preconditioner: DeviceSparsePreconditioner,
    ) -> anyhow::Result<()> {
        ensure!(
            self.lock()
                .preconditioners
                .insert(preconditioner_id, Arc::new(preconditioner))
                .is_none(),
            "sparse preconditioner identifier already exists"
        );
        Ok(())
    }

    pub(in crate::backend::wgpu::provider) fn preconditioner(
        &self,
        provider: &super::super::WgpuProvider,
        handle: &SparsePreconditionerHandle,
    ) -> anyhow::Result<Arc<DeviceSparsePreconditioner>> {
        validate_preconditioner_owner(handle, provider.runtime_device_id)?;
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

    pub(in crate::backend::wgpu::provider) fn free_pattern(
        &self,
        provider: &super::super::WgpuProvider,
        handle: &SparsePatternHandle,
    ) -> anyhow::Result<()> {
        validate_pattern_owner(handle, provider.runtime_device_id)?;
        ensure!(
            self.lock().patterns.remove(&handle.pattern_id).is_some(),
            "sparse pattern is not resident"
        );
        Ok(())
    }

    pub(in crate::backend::wgpu::provider) fn free_matrix(
        &self,
        provider: &super::super::WgpuProvider,
        handle: &SparseMatrixHandle,
    ) -> anyhow::Result<()> {
        validate_matrix_owner(handle, provider.runtime_device_id)?;
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

    pub(in crate::backend::wgpu::provider) fn free_preconditioner(
        &self,
        provider: &super::super::WgpuProvider,
        handle: &SparsePreconditionerHandle,
    ) -> anyhow::Result<()> {
        validate_preconditioner_owner(handle, provider.runtime_device_id)?;
        ensure!(
            self.lock()
                .preconditioners
                .remove(&handle.preconditioner_id)
                .is_some(),
            "sparse preconditioner is not resident"
        );
        Ok(())
    }

    pub(in crate::backend::wgpu::provider) fn resident_bytes(&self) -> u64 {
        let storage = self.lock();
        let mut patterns = HashSet::new();
        let mut total = 0_u64;
        for pattern in storage
            .patterns
            .values()
            .chain(storage.matrices.values().map(|matrix| &matrix.pattern))
        {
            let identity = Arc::as_ptr(pattern) as usize;
            if patterns.insert(identity) {
                total = total
                    .saturating_add(pattern.row_offsets.size())
                    .saturating_add(pattern.column_indices.size());
            }
        }
        let matrix_bytes = storage.matrices.values().fold(total, |bytes, matrix| {
            bytes.saturating_add(matrix.values.size())
        });
        let persistent_bytes = storage
            .preconditioners
            .values()
            .fold(matrix_bytes, |bytes, preconditioner| {
                bytes.saturating_add(preconditioner.inverse_diagonal.size())
            });
        persistent_bytes.saturating_add(self.transient_bytes.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::SparseState;

    #[test]
    fn transient_reservations_are_visible_and_release_on_drop() {
        let state = SparseState::default();
        let first = state.reserve_transient(128).expect("first reservation");
        assert_eq!(state.resident_bytes(), 128);
        {
            let second = state.reserve_transient(64).expect("second reservation");
            assert_eq!(state.resident_bytes(), 192);
            drop(second);
        }
        assert_eq!(state.resident_bytes(), 128);
        drop(first);
        assert_eq!(state.resident_bytes(), 0);
    }
}
