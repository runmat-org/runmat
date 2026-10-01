//! Native floating dimensional reductions over authenticated in-process payloads.

use super::*;
use runmat_accelerate_api::GpuTensorLayout;
use std::ops::AddAssign;

#[cfg(test)]
mod tests;

impl InProcessProvider {
    pub(super) fn native_sum_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        self.native_additive_dim(input, dim, false)
    }

    pub(super) fn native_mean_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        self.native_additive_dim(input, dim, true)
    }

    fn native_additive_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        mean: bool,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            input.shape().len() == 2,
            "reduce_sum_dim: only 2D supported"
        );
        ensure!(dim <= 1, "reduce_sum_dim: only dims 0 or 1 supported");
        ensure!(
            input.physical().storage() == GpuTensorStorage::Real,
            "reduce_sum_dim: real storage is required"
        );
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let layout = input.view().layout();
        if let GpuTensorLayout::Transpose(info) = layout {
            ensure!(
                info.base_rows == cols && info.base_cols == rows,
                "reduce_sum_dim: transpose view differs from source geometry"
            );
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("reduce_sum_dim: extent overflow"))?;
        let (output, shape) = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("reduce_sum_dim: allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("reduce_sum_dim: input is not a live owned allocation"))?;
            let shape = if dim == 0 {
                vec![1, cols]
            } else {
                vec![rows, 1]
            };
            let output = match data {
                InProcessNumericData::F32(values) => {
                    ensure!(
                        !runmat_accelerate_api::handle_is_logical(input),
                        "reduce_sum_dim: logical storage requires its canonical double payload"
                    );
                    ensure!(
                        values.len() == count,
                        "reduce_sum_dim: payload extent mismatch"
                    );
                    let mut reduced = sum_typed(values, rows, cols, dim, layout);
                    if mean {
                        for value in &mut reduced {
                            *value /= input.shape()[dim] as f32;
                        }
                    }
                    InProcessNumericData::F32(reduced)
                }
                InProcessNumericData::F64(values) => {
                    ensure!(
                        values.len() == count,
                        "reduce_sum_dim: payload extent mismatch"
                    );
                    let mut reduced = sum_typed(values, rows, cols, dim, layout);
                    if mean {
                        for value in &mut reduced {
                            *value /= input.shape()[dim] as f64;
                        }
                    }
                    InProcessNumericData::F64(reduced)
                }
                InProcessNumericData::Integer(_) => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "reduce_sum_dim: native floating storage is required",
                    ));
                }
            };
            (output, shape)
        };
        self.allocate_numeric_data(output, shape, GpuTensorStorage::Real)
    }
}

fn sum_typed<T: Copy + Default + AddAssign>(
    data: &[T],
    rows: usize,
    cols: usize,
    dim: usize,
    layout: GpuTensorLayout,
) -> Vec<T> {
    let mut result = vec![T::default(); if dim == 0 { cols } else { rows }];
    for col in 0..cols {
        for row in 0..rows {
            let source = match layout {
                GpuTensorLayout::Contiguous => row + col * rows,
                GpuTensorLayout::Transpose(_) => col + row * cols,
            };
            result[if dim == 0 { col } else { row }] += data[source];
        }
    }
    result
}
