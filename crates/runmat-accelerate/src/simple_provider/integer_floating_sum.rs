//! Ordinary integer sums accumulate in double; native saturating sums use
//! the separate typed integer reduction hooks.

use super::*;
use runmat_accelerate_api::GpuTensorLayout;

impl InProcessProvider {
    pub(super) fn floating_integer_sum(
        &self,
        input: &GpuTensorHandle,
        dim: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            input.physical().storage() == GpuTensorStorage::Real,
            "ordinary integer sum requires real storage"
        );
        let shape = input.shape();
        let count = checked_elements(shape)?;
        let layout = input.view().layout();
        if let GpuTensorLayout::Transpose(info) = layout {
            ensure!(
                shape.len() == 2 && info.base_rows == shape[1] && info.base_cols == shape[0],
                "ordinary integer sum transpose view differs from source geometry"
            );
        }
        let output_shape = if let Some(dim) = dim {
            ensure!(
                dim < shape.len(),
                "ordinary integer sum dimension out of bounds"
            );
            let mut output_shape = shape.to_vec();
            output_shape[dim] = 1;
            output_shape
        } else {
            vec![1, 1]
        };
        let output_count = checked_elements(&output_shape)?;
        let result = {
            let guard = integer_registry(self.device_id)
                .lock()
                .map_err(|_| anyhow!("ordinary integer sum owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("ordinary integer sum input is not live and owned"))?;
            ensure!(
                integer_data_len(data) == count,
                "ordinary integer sum payload extent mismatch"
            );
            let mut result = Vec::new();
            result.try_reserve_exact(output_count)?;
            result.resize(output_count, 0.0_f64);
            macro_rules! accumulate {
                ($values:expr) => {{
                    for linear in 0..count {
                        let source = physical_index(linear, shape, layout);
                        let destination = output_index(linear, shape, &output_shape, dim)?;
                        result[destination] += $values[source] as f64;
                    }
                }};
            }
            match data {
                HostIntegerDataOwned::I8(values) => accumulate!(values),
                HostIntegerDataOwned::I16(values) => accumulate!(values),
                HostIntegerDataOwned::I32(values) => accumulate!(values),
                HostIntegerDataOwned::I64(values) => accumulate!(values),
                HostIntegerDataOwned::U8(values) => accumulate!(values),
                HostIntegerDataOwned::U16(values) => accumulate!(values),
                HostIntegerDataOwned::U32(values) => accumulate!(values),
                HostIntegerDataOwned::U64(values) => accumulate!(values),
            }
            result
        };
        self.allocate_numeric_data(
            InProcessNumericData::F64(result),
            output_shape,
            GpuTensorStorage::Real,
        )
    }
}

fn checked_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| anyhow!("ordinary integer sum shape exceeds provider capacity"))
    })
}

fn physical_index(linear: usize, shape: &[usize], layout: GpuTensorLayout) -> usize {
    match layout {
        GpuTensorLayout::Contiguous => linear,
        GpuTensorLayout::Transpose(_) => {
            let row = linear % shape[0];
            let column = linear / shape[0];
            column + row * shape[1]
        }
    }
}

fn output_index(
    linear: usize,
    shape: &[usize],
    output_shape: &[usize],
    reduced: Option<usize>,
) -> Result<usize> {
    if reduced.is_none() {
        return Ok(0);
    }
    let mut remainder = linear;
    let mut stride = 1usize;
    let mut destination = 0usize;
    for (axis, &extent) in shape.iter().enumerate() {
        let coordinate = if extent == 0 {
            0
        } else {
            let coordinate = remainder % extent;
            remainder /= extent;
            coordinate
        };
        if reduced != Some(axis) {
            destination = destination
                .checked_add(
                    coordinate
                        .checked_mul(stride)
                        .ok_or_else(|| anyhow!("ordinary integer sum index overflow"))?,
                )
                .ok_or_else(|| anyhow!("ordinary integer sum index overflow"))?;
        }
        stride = stride
            .checked_mul(output_shape[axis])
            .ok_or_else(|| anyhow!("ordinary integer sum stride overflow"))?;
    }
    Ok(destination)
}

#[cfg(test)]
#[path = "integer_floating_sum/tests.rs"]
mod tests;
