//! Native floating dimensional reductions over authenticated in-process payloads.

use super::*;
use runmat_accelerate_api::GpuTensorLayout;
use std::ops::AddAssign;

#[cfg(test)]
mod tests;

impl InProcessProvider {
    pub(super) fn mean_hook_admitted(
        &self,
        input: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<bool> {
        let guard = numeric_registry()
            .lock_for(self.device_id)
            .map_err(|_| anyhow!("mean: allocation owner is poisoned"))?;
        ensure!(
            guard.get(input).is_some(),
            "mean: input is not a live owned allocation"
        );
        if let Some(dim) = dim {
            ensure!(
                dim < input.shape().len(),
                "reduce_mean_dim: dimension out of bounds"
            );
        }
        Ok(input.physical().storage() != GpuTensorStorage::ComplexInterleaved)
    }

    pub(super) fn admit_mean_storage(&self, input: &GpuTensorHandle) -> Result<()> {
        if input.physical().storage() != GpuTensorStorage::ComplexInterleaved {
            return Ok(());
        }
        let guard = numeric_registry()
            .lock_for(self.device_id)
            .map_err(|_| anyhow!("mean: allocation owner is poisoned"))?;
        ensure!(
            guard.get(input).is_some(),
            "mean: input is not a live owned allocation"
        );
        Err(runmat_accelerate_api::unsupported_provider_operation(
            "mean: complex resident reduction is not implemented by this provider",
        ))
    }

    pub(super) fn ordinary_numeric_sum(
        &self,
        input: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let shape = input.shape();
        let count = shape
            .iter()
            .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))
            .ok_or_else(|| anyhow!("sum: extent overflow"))?;
        if let Some(axis) = dim {
            ensure!(axis < shape.len(), "sum: dimension out of bounds");
        }
        let layout = input.view().layout();
        if let GpuTensorLayout::Transpose(info) = layout {
            ensure!(
                shape.len() == 2 && info.base_rows == shape[1] && info.base_cols == shape[0],
                "sum: transpose view differs from source geometry"
            );
        }
        let mut output_shape = if dim.is_some() {
            shape.to_vec()
        } else {
            vec![1, 1]
        };
        if let Some(axis) = dim {
            output_shape[axis] = 1;
        }
        let output_count = output_shape
            .iter()
            .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))
            .ok_or_else(|| anyhow!("sum: output extent overflow"))?;
        let storage = input.physical().storage();
        let lane_count = if storage == GpuTensorStorage::ComplexInterleaved {
            2
        } else {
            1
        };
        let expected = count
            .checked_mul(lane_count)
            .ok_or_else(|| anyhow!("sum: payload extent overflow"))?;
        let output_len = output_count
            .checked_mul(lane_count)
            .ok_or_else(|| anyhow!("sum: result extent overflow"))?;
        let result = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("sum: allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("sum: input is not a live owned allocation"))?;
            let mut result = Vec::new();
            result.try_reserve_exact(output_len)?;
            result.resize(output_len, 0.0_f64);
            match data {
                InProcessNumericData::F32(values) => {
                    ensure!(values.len() == expected, "sum: payload extent mismatch");
                    accumulate_sum_as_double(values, &mut result, shape, dim, layout, lane_count);
                }
                InProcessNumericData::F64(values) => {
                    ensure!(values.len() == expected, "sum: payload extent mismatch");
                    accumulate_sum_as_double(values, &mut result, shape, dim, layout, lane_count);
                }
                InProcessNumericData::Integer(_) => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "sum: floating storage is required",
                    ))
                }
            }
            result
        };
        self.allocate_numeric_data(InProcessNumericData::F64(result), output_shape, storage)
    }

    pub(super) fn native_mean_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        self.native_additive_dim(input, dim, true)
    }

    pub(super) fn native_real_sum_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            !runmat_accelerate_api::handle_is_logical(input)
                && runmat_accelerate_api::handle_integer_type(input).is_none(),
            "native real sum requires floating storage"
        );
        self.native_additive_dim(input, dim, false)
    }

    fn native_additive_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        divide_by_extent: bool,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            dim < input.shape().len(),
            "reduce_sum_dim: dimension out of bounds"
        );
        let rows = input.shape()[0];
        let cols = input.shape().get(1).copied().unwrap_or(1);
        let layout = input.view().layout();
        if let GpuTensorLayout::Transpose(info) = layout {
            ensure!(
                input.shape().len() == 2 && info.base_rows == cols && info.base_cols == rows,
                "reduce_sum_dim: transpose view differs from source geometry"
            );
        }
        self.admit_mean_storage(input)?;
        ensure!(
            input.shape().len() == 2,
            "reduce_sum_dim: only 2D supported"
        );
        ensure!(dim <= 1, "reduce_sum_dim: only dims 0 or 1 supported");
        ensure!(
            input.physical().storage() == GpuTensorStorage::Real,
            "reduce_sum_dim: real storage is required"
        );
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
                    if !divide_by_extent {
                        ensure!(
                            runmat_accelerate_api::handle_precision(input)
                                == Some(runmat_accelerate_api::ProviderPrecision::F32),
                            "native real sum: floating class differs from payload"
                        );
                    }
                    ensure!(
                        !runmat_accelerate_api::handle_is_logical(input),
                        "reduce_sum_dim: logical storage requires its canonical double payload"
                    );
                    ensure!(
                        values.len() == count,
                        "reduce_sum_dim: payload extent mismatch"
                    );
                    let mut reduced = sum_typed(values, rows, cols, dim, layout);
                    if divide_by_extent {
                        for value in &mut reduced {
                            *value /= input.shape()[dim] as f32;
                        }
                    }
                    InProcessNumericData::F32(reduced)
                }
                InProcessNumericData::F64(values) => {
                    if !divide_by_extent {
                        ensure!(
                            runmat_accelerate_api::handle_precision(input)
                                == Some(runmat_accelerate_api::ProviderPrecision::F64),
                            "native real sum: floating class differs from payload"
                        );
                    }
                    ensure!(
                        values.len() == count,
                        "reduce_sum_dim: payload extent mismatch"
                    );
                    let mut reduced = sum_typed(values, rows, cols, dim, layout);
                    if divide_by_extent {
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

fn accumulate_sum_as_double<T: Copy + Into<f64>>(
    values: &[T],
    output: &mut [f64],
    shape: &[usize],
    dim: Option<usize>,
    layout: GpuTensorLayout,
    lanes: usize,
) {
    let count = values.len() / lanes;
    let geometry = dim.map(|axis| {
        let before = shape[..axis].iter().product::<usize>();
        let span = before * shape[axis];
        (before, span)
    });
    for linear in 0..count {
        let source = match layout {
            GpuTensorLayout::Contiguous => linear,
            GpuTensorLayout::Transpose(_) => {
                let row = linear % shape[0];
                let column = linear / shape[0];
                column + row * shape[1]
            }
        };
        let destination = if let Some((before, span)) = geometry {
            (linear % before) + (linear / span) * before
        } else {
            0
        };
        if lanes == 2 {
            let real: f64 = values[source * 2].into();
            let imaginary: f64 = values[source * 2 + 1].into();
            if real.is_nan() || imaginary.is_nan() {
                output[destination * 2] = f64::NAN;
                output[destination * 2 + 1] = f64::NAN;
                continue;
            }
            output[destination * 2] += real;
            output[destination * 2 + 1] += imaginary;
        } else {
            output[destination] += values[source].into();
        }
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
