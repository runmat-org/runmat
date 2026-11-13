use anyhow::{anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::{GpuTensorHandle, GpuTensorStorage, HostNumericTensorView, NumericElementType};

use super::{SparsePatternDescriptor, SparsePatternHandle, SparsePreconditionerKind};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMatrixDescriptor {
    pub pattern_id: u64,
    pub element_type: NumericElementType,
    pub storage: GpuTensorStorage,
}

impl SparseMatrixDescriptor {
    pub fn validate(self) -> anyhow::Result<()> {
        ensure!(
            self.element_type.precision().is_some(),
            "sparse numerical operators require floating-point values"
        );
        Ok(())
    }

    pub fn checked_value_bytes(self, nnz: u64) -> Option<u64> {
        let lane_count = match self.storage {
            GpuTensorStorage::Real => nnz,
            GpuTensorStorage::ComplexInterleaved => nnz.checked_mul(2)?,
        };
        lane_count.checked_mul(u64::try_from(self.element_type.element_size()).ok()?)
    }
}

pub fn validate_sparse_values(
    pattern: &SparsePatternHandle,
    values: HostNumericTensorView<'_>,
) -> anyhow::Result<SparseMatrixDescriptor> {
    values.validate()?;
    match values.data {
        crate::HostNumericDataView::F64(data) => ensure!(
            data.iter().all(|value| value.is_finite()),
            "sparse values must be finite"
        ),
        crate::HostNumericDataView::F32(data) => ensure!(
            data.iter().all(|value| value.is_finite()),
            "sparse values must be finite"
        ),
        _ => {}
    }
    ensure!(
        values.shape
            == [usize::try_from(pattern.descriptor.nnz)
                .map_err(|_| anyhow!("sparse nonzero count does not fit host usize"))?],
        "sparse value shape must equal the pattern nonzero count"
    );
    let descriptor = SparseMatrixDescriptor {
        pattern_id: pattern.pattern_id,
        element_type: values.element_type(),
        storage: values.storage,
    };
    descriptor.validate()?;
    Ok(descriptor)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMatrixHandle {
    pub device_id: u32,
    pub matrix_id: u64,
    pub pattern: SparsePatternDescriptor,
    pub descriptor: SparseMatrixDescriptor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsePreconditionerDescriptor {
    pub matrix_id: u64,
    pub kind: SparsePreconditionerKind,
    pub dimension: u64,
    pub element_type: NumericElementType,
    pub storage: GpuTensorStorage,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsePreconditionerHandle {
    pub device_id: u32,
    pub preconditioner_id: u64,
    pub descriptor: SparsePreconditionerDescriptor,
}

pub fn validate_matrix_owner(
    matrix: &SparseMatrixHandle,
    provider_device_id: u32,
) -> anyhow::Result<()> {
    ensure!(
        matrix.device_id == provider_device_id,
        "sparse matrix belongs to a different provider"
    );
    matrix.pattern.validate()?;
    matrix.descriptor.validate()
}

pub fn validate_preconditioner_owner(
    preconditioner: &SparsePreconditionerHandle,
    provider_device_id: u32,
) -> anyhow::Result<()> {
    ensure!(
        preconditioner.device_id == provider_device_id,
        "sparse preconditioner belongs to a different provider"
    );
    Ok(())
}

pub(super) fn validate_dense_owner(
    handle: &GpuTensorHandle,
    provider_device_id: u32,
) -> anyhow::Result<()> {
    ensure!(
        handle.device_id() == provider_device_id,
        "dense vector belongs to a different provider"
    );
    ensure!(
        handle.descriptor().element_type.is_some() && handle.descriptor().storage.is_some(),
        "dense vector is missing its durable numeric descriptor"
    );
    Ok(())
}

pub(super) fn ensure_vector_matches_matrix(
    handle: &GpuTensorHandle,
    expected_length: u64,
    matrix: SparseMatrixDescriptor,
    operation: &str,
) -> anyhow::Result<()> {
    ensure!(
        handle.shape()
            == [usize::try_from(expected_length)
                .map_err(|_| anyhow!("{operation} length does not fit usize"))?],
        "{operation} shape does not match the sparse operator"
    );
    ensure!(
        handle.descriptor().element_type == Some(matrix.element_type)
            && handle.descriptor().storage == Some(matrix.storage),
        "{operation} representation does not match the sparse operator"
    );
    Ok(())
}
