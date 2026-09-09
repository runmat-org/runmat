use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    validate_matrix_owner, validate_pattern_owner, validate_sparse_values, GpuTensorStorage,
    HostCsrPatternView, HostNumericDataView, HostNumericTensorView, NumericElementType,
    SparseMatrixDescriptor, SparseMatrixHandle, SparsePatternDescriptor, SparsePatternHandle,
};

mod contract;
mod pipelines;
mod spmv;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(super) use contract::{capabilities, cost, feasibility};
pub(super) use pipelines::SparsePipelines;
pub(super) use spmv::execute as execute_spmv;

pub(super) struct DeviceCsrPattern {
    descriptor: SparsePatternDescriptor,
    row_offsets: Arc<wgpu::Buffer>,
    column_indices: Arc<wgpu::Buffer>,
}

pub(super) struct DeviceSparseMatrix {
    pattern: Arc<DeviceCsrPattern>,
    descriptor: SparseMatrixDescriptor,
    values: Arc<wgpu::Buffer>,
}

#[derive(Default)]
struct SparseStorage {
    patterns: HashMap<u64, Arc<DeviceCsrPattern>>,
    matrices: HashMap<u64, Arc<DeviceSparseMatrix>>,
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

    pub(super) fn upload_pattern(
        &self,
        provider: &super::WgpuProvider,
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
        let pattern = Arc::new(DeviceCsrPattern {
            descriptor,
            row_offsets: upload_u32(provider, "runmat-sparse-row-offsets", &row_offsets)?,
            column_indices: upload_u32(provider, "runmat-sparse-column-indices", &column_indices)?,
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

    pub(super) fn bind_values(
        &self,
        provider: &super::WgpuProvider,
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
        let matrix = Arc::new(DeviceSparseMatrix {
            pattern: resident_pattern,
            descriptor,
            values: upload_bytes(
                provider,
                "runmat-sparse-values",
                bytes,
                descriptor.element_type.element_size() as u64,
            )?,
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

    pub(super) fn matrix(
        &self,
        provider: &super::WgpuProvider,
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

    pub(super) fn free_pattern(
        &self,
        provider: &super::WgpuProvider,
        handle: &SparsePatternHandle,
    ) -> anyhow::Result<()> {
        validate_pattern_owner(handle, provider.runtime_device_id)?;
        ensure!(
            self.lock().patterns.remove(&handle.pattern_id).is_some(),
            "sparse pattern is not resident"
        );
        Ok(())
    }

    pub(super) fn free_matrix(
        &self,
        provider: &super::WgpuProvider,
        handle: &SparseMatrixHandle,
    ) -> anyhow::Result<()> {
        validate_matrix_owner(handle, provider.runtime_device_id)?;
        ensure!(
            self.lock().matrices.remove(&handle.matrix_id).is_some(),
            "sparse matrix is not resident"
        );
        Ok(())
    }

    pub(super) fn resident_bytes(&self) -> u64 {
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
        storage.matrices.values().fold(total, |bytes, matrix| {
            bytes.saturating_add(matrix.values.size())
        })
    }

    pub(super) fn clear(&self) {
        let mut storage = self.lock();
        storage.patterns.clear();
        storage.matrices.clear();
    }
}

fn numeric_element_type(precision: super::NumericPrecision) -> NumericElementType {
    match precision {
        super::NumericPrecision::F32 => NumericElementType::F32,
        super::NumericPrecision::F64 => NumericElementType::F64,
    }
}

fn storage_binding_limit(provider: &super::WgpuProvider) -> u64 {
    provider.adapter_limits.max_buffer_size.min(u64::from(
        provider.adapter_limits.max_storage_buffer_binding_size,
    ))
}

fn upload_u32(
    provider: &super::WgpuProvider,
    label: &str,
    values: &[u32],
) -> anyhow::Result<Arc<wgpu::Buffer>> {
    upload_bytes(provider, label, bytemuck::cast_slice(values), 4)
}

fn upload_bytes(
    provider: &super::WgpuProvider,
    label: &str,
    bytes: &[u8],
    minimum_size: u64,
) -> anyhow::Result<Arc<wgpu::Buffer>> {
    use wgpu::util::DeviceExt;

    let zeroes = [0_u8; 8];
    let minimum_size = usize::try_from(minimum_size)
        .map_err(|_| anyhow!("{label} minimum allocation exceeds the host address space"))?;
    ensure!(
        (1..=zeroes.len()).contains(&minimum_size),
        "{label} has an unsupported minimum allocation"
    );
    let contents = if bytes.is_empty() {
        &zeroes[..minimum_size]
    } else {
        bytes
    };
    let binding_limit = storage_binding_limit(provider);
    ensure!(
        contents.len() as u64 <= binding_limit,
        "{label} exceeds the WGPU storage-buffer binding limit"
    );
    let buffer = Arc::new(provider.device_ref().create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    ));
    provider
        .telemetry
        .record_upload_bytes(contents.len() as u64);
    Ok(buffer)
}
