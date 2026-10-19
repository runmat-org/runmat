//! Native in-process matrix multiplication over authenticated resident payloads.
use super::{numeric_registry, InProcessNumericData, InProcessProvider};
use anyhow::{anyhow, Result};
use runmat_accelerate_api::{
    unsupported_provider_operation, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage,
};
use std::ops::{AddAssign, Mul};

#[cfg(test)]
mod tests;

pub(super) fn multiply(
    provider: &InProcessProvider,
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
) -> Result<GpuTensorHandle> {
    // Authenticate both handles before interpreting their views or payloads.
    // The guard retains their leases for the duration of the computation.
    let (data, shape) = {
        let guard = numeric_registry()
            .lock_for(provider.device_id)
            .map_err(|_| anyhow!("matmul: allocation owner is poisoned"))?;
        let left = guard
            .get(a)
            .ok_or_else(|| anyhow!("matmul: left input is not a live owned allocation"))?;
        let right = guard
            .get(b)
            .ok_or_else(|| anyhow!("matmul: right input is not a live owned allocation"))?;
        let left_logical = guard
            .producer_is_logical(a)
            .ok_or_else(|| anyhow!("matmul: left input classification is unavailable"))?;
        let right_logical = guard
            .producer_is_logical(b)
            .ok_or_else(|| anyhow!("matmul: right input classification is unavailable"))?;
        if a.physical().storage() != GpuTensorStorage::Real
            || b.physical().storage() != GpuTensorStorage::Real
            || left_logical
            || right_logical
            || a.view().is_logical()
            || b.view().is_logical()
        {
            return Err(unsupported_provider_operation(
                "matmul: real nonlogical floating storage is required",
            ));
        }
        // This owner admits full-payload contiguous reshapes and materializes
        // transpose results. It does not authorize a lazy transpose view.
        if a.view().layout() != GpuTensorLayout::Contiguous
            || b.view().layout() != GpuTensorLayout::Contiguous
        {
            return Err(unsupported_provider_operation(
                "matmul: noncontiguous views require materialization",
            ));
        }
        let geometry = Geometry::new(a.shape(), b.shape())?;
        let data = match (left, right) {
            (InProcessNumericData::F32(left), InProcessNumericData::F32(right)) => {
                geometry.validate_payloads(left.len(), right.len())?;
                InProcessNumericData::F32(multiply_typed(left, right, geometry)?)
            }
            (InProcessNumericData::F64(left), InProcessNumericData::F64(right)) => {
                geometry.validate_payloads(left.len(), right.len())?;
                InProcessNumericData::F64(multiply_typed(left, right, geometry)?)
            }
            _ => {
                return Err(unsupported_provider_operation(
                    "matmul: inputs require the same native floating precision",
                ));
            }
        };
        (data, vec![geometry.rows, geometry.columns])
    };
    provider.allocate_numeric_data(data, shape, GpuTensorStorage::Real)
}

#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    inner: usize,
    columns: usize,
    left_len: usize,
    right_len: usize,
    output_len: usize,
}

impl Geometry {
    fn new(left: &[usize], right: &[usize]) -> Result<Self> {
        let ([rows, inner], [right_inner, columns]) = (left, right) else {
            return Err(unsupported_provider_operation(
                "matmul: only 2D arrays are supported",
            ));
        };
        if inner != right_inner {
            return Err(anyhow!("matmul: inner dimensions must agree"));
        }
        let left_len = rows
            .checked_mul(*inner)
            .ok_or_else(|| anyhow!("matmul: left element count overflow"))?;
        let right_len = right_inner
            .checked_mul(*columns)
            .ok_or_else(|| anyhow!("matmul: right element count overflow"))?;
        let output_len = rows
            .checked_mul(*columns)
            .ok_or_else(|| anyhow!("matmul: output element count overflow"))?;
        Ok(Self {
            rows: *rows,
            inner: *inner,
            columns: *columns,
            left_len,
            right_len,
            output_len,
        })
    }

    fn validate_payloads(self, left_len: usize, right_len: usize) -> Result<()> {
        if left_len < self.left_len || right_len < self.right_len {
            return Err(anyhow!("matmul: view exceeds authenticated payload length"));
        }
        Ok(())
    }
}

fn multiply_typed<T>(left: &[T], right: &[T], geometry: Geometry) -> Result<Vec<T>>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
{
    let mut output = Vec::new();
    output
        .try_reserve_exact(geometry.output_len)
        .map_err(|error| anyhow!("matmul: output allocation failed: {error}"))?;
    output.resize(geometry.output_len, T::default());
    for column in 0..geometry.columns {
        for row in 0..geometry.rows {
            let mut sum = T::default();
            for inner in 0..geometry.inner {
                let left_index = row + inner * geometry.rows;
                let right_index = inner + column * geometry.inner;
                sum += left[left_index] * right[right_index];
            }
            output[row + column * geometry.rows] = sum;
        }
    }
    Ok(output)
}
