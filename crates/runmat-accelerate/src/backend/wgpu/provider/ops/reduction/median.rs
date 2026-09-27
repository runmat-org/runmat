//! Host-mediated median retains native storage and the provider's zero-based axis ABI.
use super::*;
use runmat_accelerate_api::{
    unsupported_provider_operation, HostNumericDataOwned, HostNumericTensorOwned,
    NumericElementType,
};

impl WgpuProvider {
    pub(crate) async fn reduce_median_exec(
        &self,
        input: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.median_native(input, None).await
    }

    pub(crate) async fn reduce_median_dim_exec(
        &self,
        input: &GpuTensorHandle,
        axis: usize,
    ) -> Result<GpuTensorHandle> {
        self.median_native(input, Some(axis)).await
    }

    async fn median_native(
        &self,
        input: &GpuTensorHandle,
        axis: Option<usize>,
    ) -> Result<GpuTensorHandle> {
        let entry = self.get_entry_raw(input)?;
        if entry.storage != GpuTensorStorage::Real
            || input.view().is_logical()
            || !matches!(
                entry.element_type,
                NumericElementType::F32 | NumericElementType::F64
            )
        {
            return Err(unsupported_provider_operation(
                "median requires real native floating storage",
            ));
        }
        if axis.is_some_and(|axis| axis > 1 || input.shape().len() != 2) {
            return Err(unsupported_provider_operation(
                "dimensional median supports zero-based axes 0 and 1 of a matrix",
            ));
        }
        // The language's existing empty Single rule yields Double NaN. Let the
        // owning runtime apply that rule rather than silently changing class.
        if entry.len == 0 && entry.element_type == NumericElementType::F32 {
            return Err(unsupported_provider_operation(
                "empty Single median requires the owning host empty-result policy",
            ));
        }
        let host = self.download_numeric_exec(input).await?;
        let (rows, cols, axis) = match axis {
            Some(axis) => (host.shape[0], host.shape[1], axis),
            None => (host.data.len(), 1, 0),
        };
        let shape = if axis == 0 {
            vec![1, cols]
        } else {
            vec![rows, 1]
        };
        let data = match host.data {
            HostNumericDataOwned::F32(values) => HostNumericDataOwned::F32(medians(
                &values,
                rows,
                cols,
                axis,
                f32::NAN,
                f32::is_nan,
                median_f32,
            )),
            HostNumericDataOwned::F64(values) => HostNumericDataOwned::F64(medians(
                &values,
                rows,
                cols,
                axis,
                f64::NAN,
                f64::is_nan,
                compute_median_inplace,
            )),
            _ => {
                return Err(anyhow!(
                    "median native download changed admitted floating storage"
                ))
            }
        };
        self.upload_numeric_exec(
            &HostNumericTensorOwned {
                data,
                shape,
                storage: GpuTensorStorage::Real,
            }
            .as_view(),
        )
    }
}

fn median_f32(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

fn medians<T: Copy>(
    values: &[T],
    rows: usize,
    cols: usize,
    axis: usize,
    nan: T,
    is_nan: fn(T) -> bool,
    median: fn(&mut [T]) -> T,
) -> Vec<T> {
    let (count, width) = if axis == 0 {
        (cols, rows)
    } else {
        (rows, cols)
    };
    if count == 0 {
        return Vec::new();
    }
    let mut scratch = Vec::with_capacity(width);
    (0..count)
        .map(|output| {
            scratch.clear();
            for offset in 0..width {
                let index = if axis == 0 {
                    offset + output * rows
                } else {
                    output + offset * rows
                };
                let value = values[index];
                if is_nan(value) {
                    return nan;
                }
                scratch.push(value);
            }
            if scratch.is_empty() {
                nan
            } else {
                median(&mut scratch)
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "median/tests.rs"]
mod tests;
